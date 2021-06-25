"""The NCSN++ model [Song et al., 2020]"""

from typing import Tuple
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..layers.layers import (
    AttnBlock, Upsample, Downsample, ResnetBlockBigGAN, conv3x3
)

# -----------------------------------------------------------------------------

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embedding_size) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = torch.log(x)
        x_proj = x_proj[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def get_sigmas(self, x):
        return x


class PositionalSinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for time steps."""
    def __init__(self, embedding_dim, sigma_min, sigma_max, num_scales):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer(
            "sigmas",
            torch.tensor(
                np.exp(
                    np.linspace(
                        np.log(sigma_max), np.log(sigma_min), num_scales
                    )
                )
            )
        )

    def forward(self, x, max_positions = 10000):
        assert len(x.shape) == 1  # and x.dtype == tf.int32
        half_dim = self.embedding_dim // 2
        # magic number 10000 is from transformers
        emb = np.log(max_positions) / (half_dim - 1)
        # emb = math.log(2.) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=x.device) * -emb
        )
        # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
        # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (x.shape[0], self.embedding_dim)
        return emb

    def get_sigmas(self, x):
        return self.sigmas[x.long()]


def get_act(name: str):
    """Get activation function"""
    if name.lower() == "elu":
        return nn.ELU()
    elif name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif name.lower() in ["silu", "swish"]:
        return nn.SiLU()
    else:
        _available_activations = ["elu", "relu", "leakyrelu", "silu"]
        raise ValueError(
            "Invalid nonlinearity, must be one of: "
            ", ".join(_available_activations)
        )


class NCSNpp(nn.Module):
    """NCSN++ model"""
    def __init__(
        self,
        *,
        in_channels: int,
        ch: int,
        ch_mult: Tuple[int],
        num_res_blocks: int,
        resolution: int,
        attn_resolutions: Tuple[int],
        resblock_type: str = "biggan",
        nonlinearity: str = "silu",
        embedding_type: str = "fourier",
        fourier_scale: float = 16.,
        time_conditional: bool = True,
        resample_with_conv: bool = True,
        fir: bool = True,
        fir_kernel: Tuple[int] = (1, 3, 3, 1),
        weight_init_scale: float = 0.,
        scale_by_sqrt2: bool = True,
        scale_output_by_sigma: bool = True,
        dropout: float = 0.1,
        sigma_min: float = 0.01,
        sigma_max: float = 50,
        num_scales: int = 1000
    ):
        """

        Args:
            in_channels:
            ch:
            ch_mult:
            num_res_blocks:
            resolution:
            attn_resolutions:
            resblock_type:
            nonlinearity:
            embedding_type:
            fourier_scale:
            time_conditional:
            resample_with_conv:
            fir:
            fir_kernel:
            weight_init_scale:
            scale_by_sqrt2:
            scale_output_by_sigma:
            dropout:
            sigma_min:
            sigma_max:
            num_scales:
        """
        super().__init__()
        self.act = get_act(nonlinearity)
        self.scale_output_by_sigma = scale_output_by_sigma
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.embedding_type = embedding_type

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
                PositionalSinusoidalEmbedding(
                    embedding_dim=ch,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    num_scales=num_scales,
                )
            )
            embed_dim = ch
        else:
            raise ValueError(
                "Invalid embedding_type. Must be one of: 'fourier', "
                f"'positional'. Was: '{self.embedding_type}'"
            )

        # Explicitly condition on time step
        if time_conditional:
            self.time_conditioning.append(nn.Linear(embed_dim, 4 * ch))
            self.time_conditioning.append(nn.Linear(4 * ch, 4 * ch))

            # Use custom weights and bias initialization
            # for i in [-1, -2]:
            #     self.time_conditioning[i].weight.data = init_weights(
            #         self.time_conditioning[i].weight.shape
            #     )
            #     nn.init.zeros_(self.time_conditioning[i].bias)

        self.time_conditioning = nn.Sequential(*self.time_conditioning)

        _AttnBlock = functools.partial(
            AttnBlock,
            init_scale=weight_init_scale,
            scale_by_sqrt2=scale_by_sqrt2,
        )

        _Upsample = functools.partial(
            Upsample,
            with_conv=resample_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        _Downsample = functools.partial(
            Downsample,
            with_conv=resample_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if resblock_type.lower() == "ddpm":
            raise NotImplementedError()
            # resnet = functools.partial(
            #     ResnetBlockDDPM,
            #     act=self.act,
            #     dropout=dropout,
            #     init_scale=weight_init_scale,
            #     scale_by_sqrt2=scale_by_sqrt2,
            #     temb_dim=4 * ch,
            # )
        elif resblock_type.lower() == "biggan":
            _ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=self.act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=weight_init_scale,
                scale_by_sqrt2=scale_by_sqrt2,
                temb_dim=4 * ch,
            )
        else:
            raise ValueError(
                f"resblock_type must be 'biggan' or 'ddpm, was {resblock_type}"
            )

        self.conv_in = conv3x3(in_channels, ch)

        # Downsampling
        self.down = nn.ModuleList()
        current_res = resolution
        in_ch = ch
        h_channels = [ch]
        for level in range(self.num_resolutions):
            stage = nn.Module()
            stage.main = nn.ModuleList()
            stage.uses_attn = current_res in attn_resolutions
            out_ch = ch * ch_mult[level]
            for _ in range(self.num_res_blocks):
                stage.main.append(_ResnetBlock(in_ch=in_ch, out_ch=out_ch))

                if stage.uses_attn:
                    stage.main.append(_AttnBlock(channels=out_ch))

                h_channels.append(out_ch)
                in_ch = out_ch

            if level != self.num_resolutions - 1:
                #
                # stage.downsample = _Downsample(in_ch=in_ch)
                #
                stage.downsample = _ResnetBlock(in_ch=in_ch, down=True)
                current_res = current_res // 2
                h_channels.append(in_ch)

            self.down.append(stage)

        # Mid
        self.mid = nn.Sequential(
            _ResnetBlock(in_ch=in_ch),
            _AttnBlock(channels=in_ch),
            _ResnetBlock(in_ch=in_ch),
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
                #
                # stage.upsample = _Upsample(in_ch=in_ch)
                #
                stage.upsample = _ResnetBlock(in_ch=in_ch, up=True)
                current_res = current_res * 2

            self.up.append(stage)

        assert not h_channels

        self.conv_out = nn.Sequential(
            nn.GroupNorm(
                num_groups=min(in_ch // 4, 32),
                num_channels=in_ch,
                eps=1e-6
            ),
            conv3x3(in_ch, in_channels, init_scale=weight_init_scale),
        )

        _lat_res = resolution // 2**(self.num_resolutions - 1)
        print(
            f"Initialized NCSN++ model.\nInput dimensions: {in_channels} x "
            f"{resolution} x {resolution} (C x H x W)\nLatent dimensions: "
            f"{in_channels} x {_lat_res} x {_lat_res} (C x H x W)"
        )

    def forward(self, x, time_cond):

        t_emb = self.time_conditioning(time_cond)

        h = self.conv_in(x)

        # Downsample
        # Store intermediate outputs for skip-connections
        hs = []
        hs.append(h)
        for stage in self.down:
            for block in stage.main:
                if stage.uses_attn and isinstance(block, AttnBlock):
                    # AttnBlock
                    h = block(h)
                else:
                    # ResBlock
                    h = block(h, t_emb)
                    hs.append(h)

            if hasattr(stage, "downsample"):
                h = stage.downsample(h, t_emb)
                hs.append(h)

        # Mid
        h = self.mid(h)

        # Upsample
        for stage in self.up:
            for block in stage.main:
                if stage.uses_attn and isinstance(block, AttnBlock):
                    # AttnBlock
                    h = block(h)
                else:
                    # ResBlock, skip-connection via concatenation
                    h = block(torch.cat([h, hs.pop()], dim=1), t_emb)

            if hasattr(stage, "upsample"):
                h = stage.upsample(h, t_emb)

        assert not hs

        h = self.conv_out(h)

        if self.scale_output_by_sigma:
            # Get the sigma values from the time step embedding
            sigmas = self.time_conditioning[0].get_sigmas(time_cond).reshape(
                (x.shape[0],) + (1,) * (len(x.shape)-1)
            )
            h /= sigmas

        return h
