"""Contains a pl.LightningModule for training a score-based generative model.
"""

from contextlib import contextmanager

import torch
import pytorch_lightning as pl

from cutlit import instantiate_from_config
from ..modules.sde.sampling import get_sampling_fn
from ..modules.ema.ema import LitEma

# -----------------------------------------------------------------------------


class ScoreBasedModel(pl.LightningModule):
    """"""
    def __init__(
        self,
        *,
        ncsn_config,
        loss_config,
        sde_config,
        ae_config = None,
        optimizer_kwargs = None,
        lr_scheduler_kwargs = None,
        ema_kwargs = None,
        sampling_kwargs = None,
        image_key = "image",
        ckpt_path: str = None,
        ignore_keys = None,
        use_ema = True,
    ):
        """"""
        super().__init__()

        # TODO Find a cleaner way to do this
        # Handle sigma-parameters explicitly
        ncsn_sigma_params = [
            ncsn_config["params"].get("sigma_min", 0.01),
            ncsn_config["params"].get("sigma_max", 50),
            ncsn_config["params"].get("num_scales", 1000)
        ]
        sde_sigma_params = [
            sde_config["params"].get("sigma_min", 0.01),
            sde_config["params"].get("sigma_max", 50),
            sde_config["params"].get("N", 1000)
        ]

        for p1, p2, name in zip(
            ncsn_sigma_params,
            sde_sigma_params,
            ["sigma_min", "sigma_max", "N / num_scales"]
        ):
            assert p1 == p2, f"{name} parameters do not match!"

        self.score_model = instantiate_from_config(ncsn_config)
        self.loss = instantiate_from_config(loss_config)
        self.sde = instantiate_from_config(sde_config)

        self.use_ema = use_ema
        self.ema = LitEma(
            self.score_model, **(ema_kwargs if ema_kwargs is not None else {})
        )

        self.ae_model = None
        if ae_config is not None:
            self.ae_model = instantiate_from_config(ae_config)

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

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def init_from_ckpt(self, ckpt_path, ignore_keys=None):
        """

        Args:
            ckpt_path:
            ignore_keys:

        Returns:

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
    def ema_scope(self, verbose=False):
        if self.use_ema:
            self.ema.store(self.score_model.parameters())
            self.ema.copy_to(self.score_model)
            if verbose:
                print("--ema-scope: Switched to EMA weights")
        elif verbose:
            print("--ema-scope: EMA disabled, no EMA weights available")
        try:
            yield
        finally:
            if self.use_ema:
                self.ema.restore(self.score_model.parameters())
                if verbose:
                    print("--ema-scope: Restored training weights")

    def get_input(self, batch, image_key, device=None):
        """

        Args:
            batch:
            image_key:
            device:

        Returns:

        """
        # TODO If using ae_model, use its get_input method?
        x = batch[image_key]
        if len(x.shape) == 3:
            x = x[None, ...]
        x = x.permute(0, 3, 1, 2).to(
            memory_format=torch.contiguous_format
        ).float()

        if device is not None:
            x = x.to(device)

        if self.ae_model is not None:
            with torch.no_grad():
                x = self.ae_model.encode(x).sample()

        return x

    def forward(self, x):
        raise NotImplementedError(
            f"A forward method does not exist for '{self.__class__.__name__}'."
            " Use the 'sample' method to generate images."
        )

    def training_step(self, batch, batch_idx):
        """"""
        inputs = self.get_input(batch, self.image_key)

        loss = self.loss(inputs, model=self.score_model, sde=self.sde)
        self.log(
            "loss", loss,
            prog_bar=True, logger=True, on_step=True, on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """"""
        inputs = self.get_input(batch, self.image_key)

        loss = self.loss(inputs, model=self.score_model, sde=self.sde)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        """"""
        optimizer = torch.optim.Adam(
            self.score_model.parameters(),
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
            self.ema(self.score_model)

    def on_validation_start(self) -> None:
        """"""
        if self.use_ema:
            self.ema.store(self.score_model.parameters())
            self.ema.copy_to(self.score_model)

    def on_validation_end(self) -> None:
        """"""
        if self.use_ema:
            self.ema.restore(self.score_model.parameters())

    def sample(self, verbose=False, **kwargs):
        """Sample from the model.

        Args:
            **kwargs: Passed to sde.sampling.get_sampling_fn

        Returns: Randomly generated sample
        """
        # TODO What about sampling "device" (<-> lightning) ?

        sampling_fn = get_sampling_fn(
            sde=self.sde,
            # TODO Automatically set `continuous` argument depending on ...
            #      loss type?
            # continuous=self.continuous,
            **{**self.default_sampling_kwargs, **kwargs}
        )

        with torch.no_grad():
            with self.ema_scope(verbose=verbose):
                x, _ = sampling_fn(self.score_model)

            if self.ae_model is not None:
                x = self.ae_model.decode(x)

        return x

    def log_images(self, batch, **kwargs):
        """Log generated samples"""
        log = dict()

        # Generate sample
        samples = self.sample()
        log["samples"] = samples

        if self.ae_model is not None:
            # Reconstruct an image using the autoencoder
            x = self.get_input(batch, self.image_key, device=self.device)
            xrec = self.ae_model.decode(x)
            log["reconstruction"] = xrec

        return log
