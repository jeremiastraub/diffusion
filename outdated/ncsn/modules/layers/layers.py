""""""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other

    https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/unet.py#L200
    """

    def __init__(
        self,
        channels,
        *,
        num_heads=1,
        num_ch_per_head=None,
        scale_by_sqrt2=True
    ):
        """Attention block

        Args:
            channels: Input channels
            num_heads: Number of heads
            num_ch_per_head: Number of channels per head. If given,
                ``num_heads`` is ignored.
            scale_by_sqrt2: Whether to scale the output by 1/sqrt(2)
        """
        super().__init__()
        self.channels = channels
        if num_ch_per_head is None:
            self.num_heads = num_heads
        else:
            assert channels % num_ch_per_head == 0, (
                f"q,k,v channels {channels} is not divisible by "
                f"num_channels_per_head {num_ch_per_head}"
            )
            self.num_heads = channels // num_ch_per_head
        self.scale_by_sqrt2 = scale_by_sqrt2

        self.norm = nn.GroupNorm(
            num_groups=min(channels // 4, 32),
            num_channels=channels,
            eps=1e-6,
        )
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        if self.scale_by_sqrt2:
            return (x + h).reshape(b, c, *spatial) * (1. / np.sqrt(2.))
        else:
            return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """QKV attention.

    https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/unet.py#L233
    """

    def forward(self, qkv):
        """Apply QKV attention.
        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1. / np.sqrt(np.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


class Upsample(nn.Module):
    """"""
    def __init__(
        self,
        in_ch: int,
        out_ch: int = None,
        with_conv: bool = True,
    ):
        """Upsampling by a factor of 2.

        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            with_conv: If True, applies convolution after interpolation
        """
        super().__init__()
        self.in_ch = in_ch
        self.with_conv = with_conv
        out_ch = out_ch if out_ch is not None else in_ch
        if with_conv:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        else:
            assert in_ch == out_ch

    def forward(self, x):
        assert x.shape[1] == self.in_ch
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int = None,
        with_conv: bool = True,
    ):
        """Downsampling by a factor of 2.

        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            with_conv: If True, use learned convolution for downsampling.
                If False, use Average Pooling.
        """
        super().__init__()
        self.in_ch = in_ch
        out_ch = out_ch if out_ch is not None else in_ch
        if with_conv:
            self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        else:
            assert in_ch == out_ch
            self.down = nn.AvgPool2d(2)

    def forward(self, x):
        assert x.shape[1] == self.in_ch
        return self.down(x)


class ResnetBlockBigGAN(nn.Module):
    def __init__(
        self,
        act,
        in_ch: int,
        out_ch: int = None,
        emb_dim: int = None,
        use_adaptive_group_norm = False,
        dropout: float = 0.,
        scale_by_sqrt2: bool = True,
        down: bool = False,
        up: bool = False,
    ):
        """Residual block which incorporates embedded conditioning information.

        Args:
            act: The activation function
            in_ch: Number of input channels
            out_ch: Number of output channels
            emb_dim: Embedding dimensionality
            use_adaptive_group_norm: Whether to use AdaGN to incorporate
                conditioning information [Dhariwal & Nichol, 2021].
            dropout: Dropout probability
            scale_by_sqrt2: Whether to scale output by 1/sqrt(2)
            down: Whether to use this block for upsampling
            up: Whether to use this block for downsampling
        """
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        self.use_adaptive_group_norm = use_adaptive_group_norm
        self.scale_by_sqrt2 = scale_by_sqrt2
        self.up_or_down = up or down
        assert not (up and down)

        # TODO Use convolution for resampling ?

        if up:
            self.res_resample = Upsample(in_ch, with_conv=False)
            self.skip_resample = Upsample(in_ch, with_conv=False)
        elif down:
            self.res_resample = Downsample(in_ch, with_conv=False)
            self.skip_resample = Downsample(in_ch, with_conv=False)

        self.in_layers = nn.Sequential(
            nn.GroupNorm(
                num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
            ),
            act,
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            act,
            nn.Linear(
                emb_dim, 2 * out_ch if self.use_adaptive_group_norm else out_ch
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(
                num_groups=min(in_ch // 4, 32), num_channels=out_ch, eps=1e-6
            ),
            act,
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
        )

        # TODO kernel size 3 or 1 ?

        if in_ch != out_ch:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb):
        if self.up_or_down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.res_resample(h)
            x = self.skip_resample(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_adaptive_group_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1. + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layers(h + emb_out)

        if self.scale_by_sqrt2:
            return (self.skip_connection(x) + h) * (1. / np.sqrt(2))
        else:
            return self.skip_connection(x) + h
