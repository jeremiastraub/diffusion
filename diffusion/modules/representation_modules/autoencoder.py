import functools
import logging

import torch
import torch.nn as nn
import numpy as np

from diffusion.modules.utils import DiagonalGaussianDistribution, get_act
from diffusion.modules.layers import (
    GaussianFourierProjection, PositionalSinusoidalEmbedding, AttentionBlock,
    ResnetBlockBigGAN, Downsample, Upsample,
)

from main import instantiate_from_config

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class RepresentationAE(nn.Module):
    def __init__(
        self,
        *,
        embed_dim,
        enc_config,
        dec_config,
    ):
        """Autoencoder model with convolution-based encoder for representation
        learning. Note that the representation model is not trained with a
        reconstruction loss. The decoder is used for processing the
        representation for different resolutions.

        Args:
            embed_dim: Latent embedding dimensionality
            enc_config: Passed to ``main.instantiate_from_config``
            dec_config: Passed to ``main.instantiate_from_config``
        """
        super().__init__()
        self.encoder = instantiate_from_config(enc_config)
        self.decoder = instantiate_from_config(dec_config)
        self.embed_dim = embed_dim

        z_channels = enc_config["params"]["z_channels"]
        double_z = enc_config["params"].get("double_z", False)

        self.quant_conv = torch.nn.Conv2d(
            2 * z_channels if double_z else z_channels,
            2 * embed_dim,
            1,
        )
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        _in_res = enc_config['params']['resolution']
        _lat_res = _in_res // 2 ** (len(enc_config['params']['ch_mult']) - 1)
        self.latent_shape = (embed_dim, _lat_res, _lat_res)
        log.info(
            f"Initialized Representation AE model.\nInput dimensions: "
            f"{enc_config['params']['in_channels']} x {_in_res} x {_in_res} "
            f"(C x H x W)\nRepresentation dims: {embed_dim} x {_lat_res} x "
            f"{_lat_res}"
        )

    def encode(self, x, **enc_kwargs):
        h = self.encoder(x, **enc_kwargs)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, **dec_kwargs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, **dec_kwargs)
        return dec
    
    def encode_to_continuous_latent(
        self, x, sample_posterior=True, **enc_kwargs
    ):
        posterior = self.encode(x, **enc_kwargs)
        if sample_posterior:
            return posterior.sample()
        else:
            return posterior.mode()
    
    def decode_from_continuous_latent(self, z, **dec_kwargs):
        return self.decode(z, **dec_kwargs)

    def forward(
        self, input, sample_posterior=True, enc_kwargs=None, dec_kwargs=None
    ):
        enc_kwargs, dec_kwargs = enc_kwargs or {}, dec_kwargs or {}
        posterior = self.encode(input, **enc_kwargs)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, **dec_kwargs)
        return dec, posterior


class RepresentationEncoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        ch_mult,
        num_res_blocks,
        attn_resolutions,
        z_channels,
        double_z=False,
        dropout=0.,
        in_channels,
        resolution,
        act="silu",
        num_heads=1,
        num_ch_per_head=None,
        embedding_type="fourier",
        fourier_scale=16.,
        resample_with_resblock=True,
        resample_with_conv=True,
        use_adaptive_group_norm=False,
        scale_by_sqrt2=False,
        num_classes=None,
        time_conditional=False,
        fc_ch=None,
        conv_out_channels=None,
    ):
        super().__init__()
        self.act = get_act(act)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_classes = num_classes
        self.embedding_type = embedding_type

        # Time embedding and conditioning
        self._time_conditional = time_conditional
        if time_conditional:
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
        self._class_conditional = self.num_classes is not None
        if self._class_conditional:
            self.label_emb = nn.Embedding(self.num_classes, 4 * ch)

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
            _Downsample = functools.partial(_ResnetBlock, down=True)
        else:
            _Downsample = functools.partial(
                Downsample, with_conv=resample_with_conv
            )

        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Downsampling
        self.down = nn.ModuleList()
        current_res = resolution
        in_ch = ch
        h_channels = [in_ch]
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
                stage.downsample = _Downsample(in_ch=in_ch)
                current_res = current_res // 2
                h_channels.append(in_ch)

            self.down.append(stage)

        # Mid
        self.mid = nn.ModuleList(
            [
                _ResnetBlock(in_ch=in_ch),
                _AttnBlock(channels=in_ch),
                _ResnetBlock(in_ch=in_ch),
            ]
        )

        # End
        self.norm_out = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )

        self.conv_out = torch.nn.Conv2d(
            in_ch,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    @property
    def time_conditional(self):
        return self._time_conditional

    @property
    def class_conditional(self):
        return self._class_conditional

    def forward(self, x, time_cond=None, y=None):
        emb = None

        if self.time_conditional:
            assert time_cond is not None
            assert time_cond.shape == (x.shape[0],)
            emb = self.time_conditioning(time_cond)

        if self.class_conditional:
            assert y is not None, "Missing class label for class-cond. model"
            assert y.shape == (x.shape[0],)
            if emb is not None:
                emb = emb + self.label_emb(y)
            else:
                emb = self.label_emb(y)

        h = self.conv_in(x)

        # Downsample
        for stage in self.down:
            for block in stage.main:
                if stage.uses_attn and isinstance(block, AttentionBlock):
                    # AttnBlock
                    h = block(h)
                else:
                    # ResBlock
                    h = block(h, emb)

            if hasattr(stage, "downsample"):
                h = stage.downsample(h, emb)

        # Mid
        for block in self.mid:
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, emb)

        # End
        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)

        return h


class RepresentationDecoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        ch_mult,
        num_res_blocks,
        attn_resolutions,
        z_channels,
        dropout=0.,
        resolution,
        act="silu",
        num_heads=1,
        num_ch_per_head=None,
        embedding_type="fourier",
        fourier_scale=16.,
        resample_with_resblock=True,
        resample_with_conv=True,
        use_adaptive_group_norm=False,
        scale_by_sqrt2=False,
        num_classes=None,
        time_conditional=False,
        fc_ch=None,
    ):
        super().__init__()
        self.act = get_act(act)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_classes = num_classes
        self.embedding_type = embedding_type

        # Time embedding and conditioning
        self._time_conditional = time_conditional
        if time_conditional:
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
        self._class_conditional = self.num_classes is not None
        if self._class_conditional:
            self.label_emb = nn.Embedding(self.num_classes, 4 * ch)

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
            _Upsample = functools.partial(_ResnetBlock, up=True)
        else:
            _Upsample = functools.partial(
                Upsample, with_conv=resample_with_conv
            )

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch = ch * ch_mult[self.num_resolutions-1]
        current_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1, z_channels, current_res, current_res)
        log.info(
            f"Working with representation of shape {self.z_shape} = "
            f"{np.prod(self.z_shape)} dimensions."
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, in_ch, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(in_ch=in_ch, out_ch=in_ch)
        self.mid.attn_1 = _AttnBlock(channels=in_ch)
        self.mid.block_2 = _ResnetBlock(in_ch=in_ch, out_ch=in_ch)

        # Upsampling
        self.up = nn.ModuleList()
        out_ch = ch
        for level in reversed(range(self.num_resolutions)):
            stage = nn.Module()
            stage.main = nn.ModuleList()
            stage.uses_attn = current_res in attn_resolutions
            out_ch = ch * ch_mult[level]

            for _ in range(self.num_res_blocks + 1):
                stage.main.append(_ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                if stage.uses_attn:
                    stage.main.append(_AttnBlock(channels=out_ch))

                in_ch = out_ch

            if level != 0:
                stage.upsample = _Upsample(in_ch=in_ch)
                current_res = current_res * 2

            self.up.append(stage)

        # out
        self.out = nn.Sequential(
            nn.GroupNorm(
                num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
            ),
            self.act,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    @property
    def time_conditional(self):
        return self._time_conditional

    @property
    def class_conditional(self):
        return self._class_conditional

    def forward(self, z, time_cond=None, y=None):
        emb = None

        # Return a list of representations on different depths
        hs = []

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        if self.time_conditional:
            assert time_cond is not None
            assert time_cond.shape == (z.shape[0],)
            emb = self.time_conditioning(time_cond)

        if self.class_conditional:
            assert y is not None, "Missing class label for class-cond. model"
            assert y.shape == (z.shape[0],)
            if emb is not None:
                emb = emb + self.label_emb(y)
            else:
                emb = self.label_emb(y)

        # Upsample
        for stage in self.up:
            for block in stage.main:
                if stage.uses_attn and isinstance(block, AttentionBlock):
                    # AttnBlock
                    h = block(h)
                else:
                    # ResBlock
                    h = block(h, emb)

            if hasattr(stage, "upsample"):
                hs.append(h)
                h = stage.upsample(h, emb)

        # out
        h = self.out(h)

        hs.append(h)
        return hs
