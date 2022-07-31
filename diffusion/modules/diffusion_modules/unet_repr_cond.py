"""Timestep-conditional and representation-conditional Diffusion UNet"""

from typing import Tuple
import functools
import logging

import torch
import torch.nn as nn

from diffusion.modules.layers import (
    AttentionBlock, Upsample, Downsample, ResnetBlockBigGAN, zero_module,
    GaussianFourierProjection, PositionalSinusoidalEmbedding
)
from diffusion.modules.utils import get_act

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class DiffusionUNetReprCond(nn.Module):
    """UNet model for timestep-conditional estimation of a quantity in the
    Diffusion process (e.g., noise, input image, mean of the reverse process).

    Contains (BigGAN) residual blocks on each depth-level as well as
    attention blocks for certain resolutions. The residual blocks are connected
    via skip-connections. The embedded time (and class) information is injected
    as bias to each residual block. The representation conditioning is injected
    via concatentation on each depth-level.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        ch: int,
        ch_mult: Tuple[int],
        num_res_blocks: int,
        resolution: int,
        attn_resolutions: Tuple[int],
        num_heads: int = 1,
        num_ch_per_head: int = None,
        nonlinearity: str = "silu",
        embedding_type: str = "fourier",
        fourier_scale: float = 16.,
        resample_with_resblock: bool = True,
        resample_with_conv: bool = True,
        use_adaptive_group_norm: bool = False,
        scale_by_sqrt2: bool = True,
        dropout: float = 0.,
        num_classes: int = None,
        repr_ch: int = None,
        repr_ch_mult: Tuple[int] = None,
    ):
        """Representation-conditional UNet.

        Args:
            in_channels: Number of channels of the input data
            ch: Number of base channels
            ch_mult: Chanel mutliplier for each depth-level
            num_res_blocks: Number or residual blocks per depth-level and
                (up/down) branch.
            resolution: Input image resolution. Will be used to evaluate the
                attention resolutions.
            attn_resolutions: The resolutions for which attention is applied
                after each residual block.
            num_heads: Number of heads in the attention blocks. Ignored if
                ``num_ch_per_head`` is given.
            num_ch_per_head: Number of channels per attention head
            nonlinearity: The type of activation function. May be:
                elu, relu, leakyrelu, swish (silu)
            embedding_type: Time embedding type. May be: positional, fourier
            fourier_scale: Fourier scale used for Fourier-Embedding
            resample_with_resblock: Whether to use residual blocks for up/down
                sampling.
            resample_with_conv: Whether to use learned convolutions for up/down
                sampling. Ignored if ``resample_with_resblock`` is True.
            use_adaptive_group_norm: Whether to use AdaGN to incorporate
                conditioning information in residual blocks
                [Dhariwal & Nichol, 2021].
            scale_by_sqrt2: Whether to scale output of residual and attention
                blocks by 1/sqrt(2).
            dropout: Dropout probability
            num_classes: if specified, then this model will be
                class-conditional with ``num_classes`` classes.
            repr_ch:
            repr_ch_mult:
        """
        super().__init__()
        self.act = get_act(nonlinearity)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.embedding_type = embedding_type
        self.num_classes = num_classes

        # Time embedding and conditioning
        self.time_conditioning = nn.ModuleList()

        if self.embedding_type.lower() == "fourier":
            self.time_conditioning.append(
                GaussianFourierProjection(
                    embedding_size=ch, scale=fourier_scale
                )
            )
            embed_dim = 2 * ch
        elif self.embedding_type.lower() == "positional":
            self.time_conditioning.append(
                PositionalSinusoidalEmbedding(embedding_dim=ch)
            )
            embed_dim = ch
        else:
            raise ValueError(
                "Invalid embedding_type. Must be one of: 'fourier', "
                f"'positional'. Was: '{self.embedding_type}'"
            )

        # Explicitly condition on time step
        self.time_conditioning.append(nn.Linear(embed_dim, 4 * ch))
        self.time_conditioning.append(self.act)
        self.time_conditioning.append(nn.Linear(4 * ch, 4 * ch))
        self.time_conditioning = nn.Sequential(*self.time_conditioning)

        # Class label embedding
        if self.class_conditional:
            self.label_emb = nn.Embedding(self.num_classes, 4 * ch)

        repr_ch = repr_ch or ch
        repr_ch_mult = repr_ch_mult or ch_mult

        _AttnBlock = functools.partial(
            AttentionBlock,
            num_heads=num_heads,
            num_ch_per_head=num_ch_per_head,
            scale_by_sqrt2=scale_by_sqrt2,
        )
        _ResnetBlock = functools.partial(
            ResnetBlockBigGAN,
            act=self.act,
            emb_dim=4 * ch,
            dropout=dropout,
            use_adaptive_group_norm=use_adaptive_group_norm,
            scale_by_sqrt2=scale_by_sqrt2,
        )

        if resample_with_resblock:
            _Upsample = functools.partial(
                _ResnetBlock, up=True
            )
            _Downsample = functools.partial(
                _ResnetBlock, down=True
            )
        else:
            _Upsample = functools.partial(
                Upsample, with_conv=resample_with_conv
            )
            _Downsample = functools.partial(
                Downsample, with_conv=resample_with_conv
            )

        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Downsampling
        self.down = nn.ModuleList()
        current_res = resolution
        in_ch = ch
        repr_in_ch = repr_ch
        out_ch = ch
        h_channels = [in_ch]
        for level in range(self.num_resolutions):
            stage = nn.Module()
            stage.main = nn.ModuleList()
            stage.uses_attn = current_res in attn_resolutions
            repr_in_ch = repr_ch * repr_ch_mult[level]
            out_ch = ch * ch_mult[level]
            if self.num_res_blocks > 0:
                stage.main.append(
                    _ResnetBlock(in_ch=in_ch + repr_in_ch, out_ch=out_ch)
                )
                if stage.uses_attn:
                    stage.main.append(_AttnBlock(channels=out_ch))

                h_channels.append(out_ch)
                in_ch = out_ch
            for _ in range(self.num_res_blocks - 1):
                stage.main.append(_ResnetBlock(in_ch=in_ch, out_ch=out_ch))

                if stage.uses_attn:
                    stage.main.append(_AttnBlock(channels=out_ch))

                h_channels.append(out_ch)
                in_ch = out_ch

            if level != self.num_resolutions - 1:
                stage.downsample = _Downsample(in_ch=in_ch)
                current_res = current_res // 2
                h_channels.append(in_ch)

            self.down.append(stage)

        # Mid
        self.mid = nn.ModuleList(
            [
                _ResnetBlock(in_ch=in_ch + repr_in_ch, out_ch=out_ch),
                _AttnBlock(channels=in_ch),
                _ResnetBlock(in_ch=in_ch, out_ch=out_ch),
            ]
        )

        # Upsampling
        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            stage = nn.Module()
            stage.main = nn.ModuleList()
            stage.uses_attn = current_res in attn_resolutions
            out_ch = ch * ch_mult[level]

            for _ in range(self.num_res_blocks + 1):
                stage.main.append(
                    _ResnetBlock(
                        in_ch=in_ch + h_channels.pop(), out_ch=out_ch
                    )
                )
                if stage.uses_attn:
                    stage.main.append(_AttnBlock(channels=out_ch))

                in_ch = out_ch

            if level != 0:
                stage.upsample = _Upsample(in_ch=in_ch)
                current_res = current_res * 2

            self.up.append(stage)

        assert not h_channels

        self.conv_out = nn.Sequential(
            nn.GroupNorm(
                num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
            ),
            self.act,
            zero_module(nn.Conv2d(in_ch, in_channels, 3, padding=1)),
        )

        _lat_res = resolution // 2**(self.num_resolutions - 1)
        log.info(
            f"Initialized Diffusion UNet.\nInput dimensions: {in_channels} x "
            f"{resolution} x {resolution} (C x H x W)\nLatent dimensions: "
            f"{int(ch * ch_mult[-1])} x {_lat_res} x {_lat_res} (C x H x W)"
        )

    @property
    def repr_conditional(self):
        return True

    @property
    def class_conditional(self):
        return self.num_classes is not None

    def forward(self, x, time_cond, repr, y=None):
        """Apply the model to an input batch

        Args:
            x: Input batch of shape (B, C, H, W)
            time_cond: time conditioning
            repr: latent representations (from deep to shallow)
            y: (optional) class conditioning information

        Returns: Estimated quantity; same shape as the input data
        """
        assert (y is None) == (self.num_classes is None), (
            "must specify y if and only if the model is class-conditional"
        )

        emb = self.time_conditioning(time_cond)

        if self.class_conditional:
            assert y is not None, "Missing class label for class-cond. model"
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = self.conv_in(x)

        # Downsample
        # Store intermediate outputs for skip-connections
        hs = []
        hs.append(h)
        for stage, r in zip(self.down, reversed(repr)):
            concatenated_repr = False
            for block in stage.main:
                if stage.uses_attn and isinstance(block, AttentionBlock):
                    # AttnBlock
                    h = block(h)
                elif not concatenated_repr:
                    # ResBlock
                    h = block(torch.cat([h, r], dim=1), emb)
                    hs.append(h)
                    concatenated_repr = True
                else:
                    h = block(h, emb)
                    hs.append(h)

            if hasattr(stage, "downsample"):
                h = stage.downsample(h, emb)
                hs.append(h)

        # Mid
        concatenated_repr = False
        for block in self.mid:
            if isinstance(block, AttentionBlock):
                h = block(h)
            elif not concatenated_repr:
                h = block(torch.cat([h, repr[0]], dim=1), emb)
                concatenated_repr = True
            else:
                h = block(h, emb)

        # Upsample
        for stage in self.up:
            for block in stage.main:
                if stage.uses_attn and isinstance(block, AttentionBlock):
                    # AttnBlock
                    h = block(h)
                else:
                    # ResBlock, skip-connection via concatenation
                    h = block(torch.cat([h, hs.pop()], dim=1), emb)

            if hasattr(stage, "upsample"):
                h = stage.upsample(h, emb)

        assert not hs
        h = self.conv_out(h)
        return h
