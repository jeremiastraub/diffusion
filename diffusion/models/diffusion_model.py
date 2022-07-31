"""Contains a pl.LightningModule for Generative Diffusion Models
"""
import logging
from contextlib import contextmanager

import torch
import pytorch_lightning as pl

from main import instantiate_from_config
from diffusion.modules.sde import get_sampler
from diffusion.modules.ema import LitEma
from diffusion.modules.losses import SDELoss
from diffusion.modules.utils import (
    RunningStd, StepSamplingSchedule, DiagonalGaussianDistribution
)
from diffusion.modules.warp import random_tps
from diffusion.modules.representation_modules import RepresentationStyleCondAE

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        *,
        diffusion_model_config,
        loss_config,
        sde_config,
        continuous: bool,
        ae_config = None,
        optimizer_kwargs = None,
        lr_scheduler_kwargs = None,
        ema_kwargs = None,
        sampling_kwargs = None,
        image_key = "image",
        cond_key = "class_label",
        representation_ae_config = None,
        representation_kl_weight: float = 1e-5,
        fix_representation_ae: bool = False,
        ckpt_path: str = None,
        ignore_keys = None,
        use_ema: bool = True,
        n_batches_running_std: int = 0,
        step_schedule_kwargs = None,
        warp_kwargs = None,
    ):
        """A generative diffusion model (DM). Can be trained in the latent
        space of a pretrained autoencoder (LDM). Can be trained jointly with
        a representation encoder (LRDM).

        All *_config arguments are passed to main.instantiate_from_config.

        Args:
            diffusion_model_config: Configuration of the diffusion model
            loss_config: Loss configuration
            sde_config: SDE configuration
            continuous: Whether to train with a continuous timestep range
                or with discrete timesteps (defined through ``sde_config``)
            ae_config: Autoencoder configuration. Make sure to specify the
                ckpt_path kwarg. The autoencoder model is not trained.
            optimizer_kwargs: Passed to torch.optim.Adam
            lr_scheduler_kwargs: Configures a learning rate scheduler
            ema_kwargs: Passed to LitEma
            sampling_kwargs: Default sampling kwargs, passed to get_sampling_fn
            image_key: The key with which to retreive the input image from a
                batch.
            cond_key: The key with which to retreive the conditioning
                information from a batch.
            representation_ae_config: Configures the latent representation
                model. Note that the decoder does not recover the encoder input
                but embeds the latent respresentation. The embedded
                representation is then passed to the diffusion model as
                conditioning information.
            representation_kl_weight: Weight of the KL-loss of the latent
                representation relative to that of the generative model.
            fix_representation_ae: If True, set requires_grad to False for
                representation AE parameters.
            ckpt_path: Restore model parameters from a checkpoint file
            ignore_keys: If ckpt_path given, ignores these state_dict keys
            use_ema: Whether to use EMA for diffusion model parameters
            n_batches_running_std: Number of batches used for estimating the
                standard deviation of the data. All data is rescaled to
                std=1 based on this (online) estimate.
            step_schedule_kwargs: Configures a timestep schedule. If None,
                the timestep is always sampled uniformly.
            warp_kwargs: Passed to ``random_tps``. Used for style-shape
                separated representation learning.
        """
        super().__init__()
        self.warp_kwargs = warp_kwargs or {}
        self.diff_model = instantiate_from_config(diffusion_model_config)
        self.loss = instantiate_from_config(loss_config)
        self.sde = instantiate_from_config(sde_config)
        self.continuous = continuous

        if self.continuous and not isinstance(self.loss, SDELoss):
            raise ValueError(
                "Continuous training is only available with the SDELoss."
            )

        self.ae_model = None
        if ae_config is not None:
            self.ae_model = instantiate_from_config(ae_config)
            # Turn off gradient computation for the ae model
            for param in self.ae_model.parameters():
                param.requires_grad = False

        self.repr_ae = None
        if representation_ae_config is not None:
            self.repr_ae = instantiate_from_config(representation_ae_config)

        self.repr_kl_weight = representation_kl_weight
        self.fix_representation_ae = fix_representation_ae

        self.use_ema = use_ema
        self.ema_context_active = False
        if use_ema:
            self.ema = LitEma(
                self.diff_model,
                **(ema_kwargs if ema_kwargs is not None else {})
            )

        self.step_schedule = StepSamplingSchedule(
            continuous=self.continuous,
            N=self.sde.N,
            T=self.sde.T,
            **(step_schedule_kwargs or {})
        )

        self.running_std = RunningStd()
        self.n_batches_running_std = n_batches_running_std

        self.register_buffer(
            "scale_factor", torch.tensor(1., dtype=torch.float32)
        )
        self.register_buffer(
            "inverse_scale_factor", torch.tensor(1., dtype=torch.float32)
        )

        self.optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {}
        )
        self.lr_scheduler_kwargs = (
            lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
        )
        self.default_sampling_kwargs = (
            sampling_kwargs if sampling_kwargs is not None else {}
        )

        self.image_key = image_key
        self.cond_key = cond_key

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

        if self.repr_ae is not None and self.fix_representation_ae:
            for param in self.repr_ae.parameters():
                param.requires_grad = False

    def init_from_ckpt(self, ckpt_path, ignore_keys=None):
        """Restores weights from a checkpoint file.

        Args:
            ckpt_path: Path to checkpoint file
            ignore_keys: Iterable containing state_dict keys not to be loaded
        """
        if ignore_keys is None:
            ignore_keys = []
        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        keys = list(state.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del state[k]
        self.load_state_dict(state, strict=False)
        print(f"Restored from {ckpt_path}")

    @contextmanager
    def ema_scope(self):
        if self.ema_context_active:
            log.debug("--ema-scope: EMA scope already active")
            yield

        elif self.use_ema:
            self.ema_context_active = True
            self.ema.store(self.diff_model.parameters())
            self.ema.copy_to(self.diff_model)
            log.debug("--ema-scope: Switched to EMA weights")
            try:
                yield
            finally:
                self.ema.restore(self.diff_model.parameters())
                self.ema_context_active = False
                log.debug("--ema-scope: Restored training weights")

        else:
            log.debug("--ema-scope: EMA disabled, no EMA weights available")
            yield

    def update_N(self, N):
        """"""
        self.sde.update_N(N)

    def input_rescaling(self, z):
        """Apply rescaling which transforms latents to std=1"""
        return z * self.scale_factor

    def inverse_input_rescaling(self, z):
        """Apply inverse rescaling which transforms to the original latents"""
        return z * self.inverse_scale_factor

    def encode(self, x, ae_sample=True):
        z = x if self.ae_model is None else self.ae_model.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample() if ae_sample else z.mode()

        z = self.input_rescaling(z)
        return z

    def decode(self, z):
        x = self.inverse_input_rescaling(z)
        if self.ae_model is not None:
            x = self.ae_model.decode(x)
        return x

    def get_input(
        self,
        batch,
        image_key,
        cond_key=None,
        device=None,
        encode=True,
    ):
        """"""
        if cond_key is None:
            y = None
        else:
            y = batch[cond_key]

        x = batch[image_key]
        if len(x.shape) == 3:
            x = x[None, ...]

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device or self.device)
        else:
            x = x.to(device or self.device)

        x = x.permute(0, 3, 1, 2).to(
            memory_format=torch.contiguous_format
        ).float()

        if not encode:
            return x, y

        return self.encode(x), y

    def forward(self, **kwargs):
        raise NotImplementedError(
            f"A forward method does not exist for '{self.__class__.__name__}'."
            " Use the 'sample' method to generate images."
        )

    def compute_loss(self, batch, *, is_train=True):
        """"""
        inputs, conditioning = self.get_input(
            batch,
            self.image_key,
            cond_key=(
                self.cond_key if self.diff_model.class_conditional else None
            ),
            encode=False,
        )

        # Apply first-stage model and data rescaling
        z = self.encode(inputs)

        # Timestep selection
        t = self.step_schedule.eval(
            self.global_step, num_samples=inputs.shape[0], device=self.device
        )

        repr_kl_loss = 0.
        repr = None
        # Latent representation KL loss
        if self.repr_ae is not None:
            enc_kwargs = {}
            dec_kwargs = {}
            if self.repr_ae.encoder.time_conditional:
                enc_kwargs["time_cond"] = t
            if self.repr_ae.encoder.class_conditional:
                enc_kwargs["y"] = conditioning
            if self.repr_ae.decoder.time_conditional:
                dec_kwargs["time_cond"] = t
            if self.repr_ae.decoder.class_conditional:
                dec_kwargs["y"] = conditioning
            if isinstance(self.repr_ae, RepresentationStyleCondAE):
                dec_kwargs["z_w"] = self.encode(
                    random_tps(inputs, **self.warp_kwargs)
                )

            repr, r_posterior = self.repr_ae(
                z, enc_kwargs=enc_kwargs, dec_kwargs=dec_kwargs
            )
            repr_kl_loss = torch.mean(r_posterior.kl())

        optional_model_kwargs = {"repr": repr} if repr is not None else {}

        # Generative diffusion model loss
        diffusion_loss = self.loss(
            z,
            model=self.diff_model,
            sde=self.sde,
            t=t,
            y=conditioning,
            **optional_model_kwargs
        )

        # Combine the two weighted losses
        loss = diffusion_loss + self.repr_kl_weight * repr_kl_loss

        if is_train:
            self.log("representation KL loss", repr_kl_loss, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        """"""
        loss = self.compute_loss(batch)
        self.log(
            "loss", loss,
            prog_bar=True, logger=True, on_step=True, on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """"""
        loss = self.compute_loss(batch, is_train=False)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        """"""
        optimizer = torch.optim.Adam(
            (
                list(self.diff_model.parameters())
                + (
                    [] if self.repr_ae is None or self.fix_representation_ae
                    else list(self.repr_ae.parameters())
                )
            ),
            lr=self.learning_rate,
            **self.optimizer_kwargs
        )

        optim_dict = {"optimizer": optimizer}
        if self.lr_scheduler_kwargs:
            optim_dict["lr_scheduler"] = self.lr_scheduler_kwargs

        return optim_dict

    def on_before_zero_grad(self, optimizer) -> None:
        """"""
        # Update EMA model parameters
        if self.use_ema:
            self.ema(self.diff_model)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
        """"""
        if self.running_std.running_n_batches < self.n_batches_running_std:
            z, conditioning = self.get_input(batch, self.image_key)
            self.running_std(z)
            std = self.running_std.std
            self.scale_factor.fill_(1. / std)
            self.inverse_scale_factor.fill_(std)

    # For newer pl version
    # def on_validation_start(self) -> None:
    #     """"""
    #     if self.use_ema:
    #         self.ema.store(self.diff_model.parameters())
    #         self.ema.copy_to(self.diff_model)
    #
    # def on_validation_end(self) -> None:
    #     """"""
    #     if self.use_ema:
    #         self.ema.restore(self.diff_model.parameters())

    def on_validation_epoch_start(self) -> None:
        """"""
        if self.use_ema:
            self.ema.store(self.diff_model.parameters())
            self.ema.copy_to(self.diff_model)

    def on_validation_epoch_end(self) -> None:
        """"""
        if self.use_ema:
            self.ema.restore(self.diff_model.parameters())

    def log_images(
        self,
        batch,
        *,
        to_log=("samples",),
        sampling_kwargs=None,
        **kwargs
    ):
        """Log images.

        Args:
            batch: current batch
            to_log: Iterable containing the image logging keys
            sampling_kwargs: Passed to ``self.sample``
            **kwargs: unused, but required for compatibility reasons

        Returns: log dict containing images keyed by logging key
        """
        _available_keys = [
            "samples", "reconstructions", "inputs", "ae_reconstructions"
        ]
        invalid_keys = [k for k in to_log if k not in _available_keys]
        if invalid_keys:
            raise ValueError(
                "Received invalid image logging keys: "
                f"{', '.join(invalid_keys)}"
                f"\nAvailable keys: {', '.join(_available_keys)}"
            )

        log = dict()

        if "samples" in to_log:
            samples = self.sample(**(sampling_kwargs or {}))
            log["samples"] = samples

        if "inputs" in to_log:
            log["inputs"] = self.get_input(
                batch, self.image_key, encode=False
            )[0]

        if "ae_reconstructions" in to_log:
            if self.ae_model is not None:
                x, _ = self.get_input(
                    batch, self.image_key, device=self.device
                )
                log["reconstructions"] = self.decode(x)
            else:
                raise ValueError(
                    "reconstructions are only available when using an ae_model"
                )

        if "reconstructions" in to_log:
            raise NotImplementedError()

        return log

    def forward_diffuse(self, batch, t, noise=None):
        """Transform an input batch to a certain point in time of the diffusion
        process.
        """
        z = torch.randn_like(batch) if noise is None else noise
        mean, std = self.sde.marginal_prob(batch, t)
        perturbed = mean + std[:, None, None, None] * z
        return perturbed

    @torch.no_grad()
    def prepare_sampling(
        self,
        *,
        sampling_method: str=None,
        num_samples: int=None,
        model_kwargs=None,
        **sampling_kwargs
    ):
        """Prepare sampling.

        Args:
            sampling_method: Name of the sampling method
            num_samples: Number of samples. Overwrites the ``shape`` entry in
                ``sampling_kwargs``.
            model_kwargs: Passed to the diffusion model
            sampling_kwargs: Passed to ``get_sampler``

        Returns: sampler, model_kwargs
        """
        sampling_kwargs = {**self.default_sampling_kwargs, **sampling_kwargs}
        default_method = sampling_kwargs.pop("sampling_method", None)
        sampling_method = sampling_method or default_method

        if "shape" not in sampling_kwargs:
            raise ValueError(
                "'shape' entry missing in 'sampling_kwargs'. Specifies the "
                "diffusion model sampling shape in (B, C, H, W) format."
            )
        if num_samples is not None:
            sampling_kwargs["shape"] = (
                [num_samples] + sampling_kwargs["shape"][1:]
            )

        sampler = get_sampler(
            name=sampling_method,
            model=self,
            **sampling_kwargs
        )

        model_kwargs = model_kwargs if model_kwargs is not None else {}

        # Prepare class conditioning
        class_idx = None
        if self.diff_model.class_conditional:
            if "y" in model_kwargs:
                if type(model_kwargs["y"]) is int:
                    assert model_kwargs["y"] < self.diff_model.num_classes
                    class_idx = torch.tensor(
                        sampling_kwargs["shape"][0]*[model_kwargs["y"]],
                        device=self.device,
                    )
            else:
                class_idx = torch.randint(
                    self.diff_model.num_classes,
                    size=(sampling_kwargs["shape"][0],),
                    device=self.device,
                )

        if class_idx is not None:
            model_kwargs["y"] = class_idx

        # Prepare representation conditioning
        if (
            self.diff_model.repr_conditional
            and "repr" not in model_kwargs
            and "repr_seq" not in model_kwargs
            and not self.repr_ae.decoder.time_conditional
        ):
            dec_kwargs = {}
            if self.repr_ae.decoder.class_conditional:
                dec_kwargs["y"] = class_idx

            lat_repr = torch.randn(
                size=(
                    (sampling_kwargs["shape"][0],) + self.repr_ae.latent_shape
                ),
                device=self.device,
            )
            model_kwargs["repr"] = self.repr_ae.decode(lat_repr, **dec_kwargs)

        return sampler, model_kwargs

    def sample(self, *, num_scales=None, **sampling_kwargs):
        """"""
        if (
            num_scales is not None
            and num_scales != self.sde.N
            and not self.continuous
        ):
            raise NotImplementedError(
                "Can't specify custom 'num_scales' for sampling from a "
                "non-continuously trained model"
            )

        with self.ema_scope():
            with self.sde.use_N(num_scales or self.sde.N):

                sampler, model_kwargs = self.prepare_sampling(
                    **sampling_kwargs
                )

                z = sampler(**model_kwargs)
                x = self.decode(z)

        return x

    def sampling_generator(
        self, *, num_scales=None, last_only=False, **sampling_kwargs
    ):
        """"""
        if (
            num_scales is not None
            and num_scales != self.sde.N
            and not self.continuous
        ):
            raise NotImplementedError(
                "Can't specify custom 'num_scales' for sampling from a "
                "non-continuously trained model"
            )

        with self.ema_scope():
            with self.sde.use_N(num_scales or self.sde.N):

                sampler, model_kwargs = self.prepare_sampling(
                    **sampling_kwargs
                )

                for i, (z, estimate) in enumerate(
                    sampler.sampling_progression(**model_kwargs)
                ):
                    if last_only and i < self.sde.N - 1:
                        yield None, None
                    else:
                        yield self.decode(z), estimate
