"""Various sampling methods"""
import abc

import torch
import numpy as np
from scipy import integrate

from diffusion.modules.sde import VESDE, VPSDE, subVPSDE
from diffusion.modules.utils import (
    from_flattened_numpy, to_flattened_numpy, get_score_fn
)

# -----------------------------------------------------------------------------

_SAMPLERS = {}
_CORRECTORS = {}
_PREDICTORS = {}


def register_sampler(cls=None, *, name: str=None):
    """A decorator for registering sampler classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _SAMPLERS:
            raise ValueError(
                f"Already registered sampler with name: {local_name}"
            )
        _SAMPLERS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_predictor(cls=None, *, name: str=None):
    """A decorator for registering predictor classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f"Already registered model with name: {local_name}"
            )
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name: str=None):
    """A decorator for registering corrector classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(
                f"Already registered model with name: {local_name}"
            )
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name: str, **init_kwargs):
    try:
        return _PREDICTORS[name.lower()](**init_kwargs)
    except KeyError as err:
        raise ValueError(
            f"'{name}' does not correspond to a registered predictor. "
            f"Available predictors: {', '.join(list(_PREDICTORS.keys()))}"
        ) from err


def get_corrector(name: str, **init_kwargs):
    try:
        return _CORRECTORS[name.lower()](**init_kwargs)
    except KeyError as err:
        raise ValueError(
            f"'{name}' does not correspond to a registered corrector. "
            f"Available correctors: {', '.join(list(_CORRECTORS.keys()))}"
        ) from err


def get_sampler(name: str, **init_kwargs):
    try:
        return _SAMPLERS[name.lower()](**init_kwargs)
    except KeyError as err:
        raise ValueError(
            f"'{name}' does not correspond to a registered sampler. "
            f"Available samplers: {', '.join(list(_SAMPLERS.keys()))}"
        ) from err


# -----------------------------------------------------------------------------


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, diff_model):
        super().__init__()
        self.sde = sde
        self.diff_model = diff_model

    @abc.abstractmethod
    def update(self, x, t, **model_kwargs):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            **model_kwargs: Passed on to the model call

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise.
                Useful for denoising.
            model_output: The direct output of the diffusion model.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, snr, n_steps, diff_model):
        super().__init__()
        self.sde = sde
        self.diff_model = diff_model
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update(self, x, t, **model_kwargs):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.
            **model_kwargs: Passed on to the model call

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise.
                Useful for denoising.
            model_output: The direct output of the diffusion model
        """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, diff_model, continuous, probability_flow=False):
        super().__init__(sde, diff_model)
        self.score_fn = get_score_fn(sde, diff_model, continuous)
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(self.score_fn, probability_flow)

    def update(self, x, t, **model_kwargs):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t, **model_kwargs)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean, None


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, diff_model, continuous, probability_flow=False):
        super().__init__(sde, diff_model)
        self.score_fn = get_score_fn(sde, diff_model, continuous)
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(self.score_fn, probability_flow)

    def update(self, x, t, **model_kwargs):
        f, G = self.rsde.discretize(x, t, **model_kwargs)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean, None


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, diff_model, continuous):
        super().__init__(sde, diff_model)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        self.score_fn = get_score_fn(sde, diff_model, continuous)

    def vesde_update(self, x, t, **model_kwargs):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep].to(t.device)
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            sde.discrete_sigmas.to(t.device)[timestep - 1],
        )
        score = self.score_fn(x, t, **model_kwargs)
        x_mean = x + score*(sigma**2 - adjacent_sigma**2)[:, None, None, None]
        std = torch.sqrt(
            (adjacent_sigma**2 * (sigma**2 - adjacent_sigma**2)) / (sigma**2)
        )
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean, score

    def vpsde_update(self, x, t, **model_kwargs):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep, None, None, None]
        score = self.score_fn(x, t, **model_kwargs)
        eps = - score * sde.sqrt_1m_alphas_cumprod.to(
            t.device
        )[timestep, None, None, None]
        x_mean = (x + beta * score) / torch.sqrt(1. - beta)
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta) * noise
        return x, x_mean, eps

    def update(self, x, t, **model_kwargs):
        if isinstance(self.sde, VESDE):
            return self.vesde_update(x, t, **model_kwargs)
        elif isinstance(self.sde, VPSDE):
            return self.vpsde_update(x, t, **model_kwargs)


@register_predictor(name='ddpm_eps_sampling')
class DDPMEpsilonSamplingPredictor(Predictor):
    """The DDPM epsilon sampling predictor."""

    def __init__(self, sde, diff_model):
        super().__init__(sde, diff_model)
        if not isinstance(sde, VPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not supported."
            )

    def update(self, x, t, **model_kwargs):
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        beta = self.sde.discrete_betas.to(t.device)[timestep, None, None, None]
        eps = self.diff_model(x, timestep, **model_kwargs)
        x_mean = (
            x - eps * beta
            / self.sde.sqrt_1m_alphas_cumprod.to(t.device)[timestep, None, None, None]
        ) / torch.sqrt(1. - beta)
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta) * noise
        return x, x_mean, eps


@register_predictor(name='ddpm_mu_sampling')
class DDPMMeanSamplingPredictor(Predictor):
    """The DDPM mean sampling predictor."""

    def __init__(self, sde, diff_model):
        super().__init__(sde, diff_model)
        if not isinstance(sde, VPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not supported."
            )

    def update(self, x, t, **model_kwargs):
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        x_mean = self.diff_model(x, timestep, **model_kwargs)
        beta = self.sde.discrete_betas.to(t.device)[timestep, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta) * noise
        return x, x_mean, x_mean


@register_predictor(name='ddpm_x0_sampling')
class DDPMImageSamplingPredictor(Predictor):
    """The DDPM image sampling predictor."""

    def __init__(self, sde, diff_model):
        super().__init__(sde, diff_model)
        if not isinstance(sde, VPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not supported."
            )

    def update(self, x, t, **model_kwargs):
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        x0 = self.diff_model(x, timestep, **model_kwargs)
        beta = self.sde.discrete_betas.to(x.device)[timestep, None, None, None]
        alpha = self.sde.alphas.to(x.device)[timestep, None, None, None]
        alphas_cumprod = self.sde.alphas_cumprod.to(x.device)[timestep, None, None, None]
        sqrt_alphas_cumprod = self.sde.sqrt_alphas_cumprod.to(x.device)[timestep, None, None, None]
        x0_weighting = sqrt_alphas_cumprod * beta / (torch.sqrt(alpha) * (1. - alphas_cumprod))
        x_weighting = torch.sqrt(alpha) * (1. - alphas_cumprod / alpha) / (1. - alphas_cumprod)
        x_mean = x0_weighting * x0 + x_weighting * x
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta) * noise
        return x, x_mean, x0


@register_predictor(name='ddim_x0_sampling')
class DDIMImageSamplingPredictor(Predictor):
    """The DDIM image sampling predictor."""

    def __init__(self, sde, diff_model):
        super().__init__(sde, diff_model)
        if not isinstance(sde, VPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not supported."
            )

    def update(self, x, t, **model_kwargs):
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        x0 = self.diff_model(x, timestep, **model_kwargs)
        alpha = self.sde.alphas.to(x.device)[timestep, None, None, None]
        alphas_cumprod = self.sde.alphas_cumprod.to(x.device)[timestep, None, None, None]
        sqrt_alphas_cumprod = self.sde.sqrt_alphas_cumprod.to(x.device)[timestep, None, None, None]

        factor = (
            torch.sqrt(1. - alphas_cumprod / alpha)
            / torch.sqrt(1. - alphas_cumprod)
        )
        x_mean = (
            torch.sqrt(alphas_cumprod / alpha) * x0
            + factor * (x - sqrt_alphas_cumprod * x0)
        )
        x = x_mean
        return x, x_mean, x0


@register_predictor(name='ddim_encoding')
class DDIMImageEncodingPredictor(Predictor):
    """The DDIM image encoding predictor."""

    def __init__(self, sde, diff_model):
        super().__init__(sde, diff_model)
        if not isinstance(sde, VPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not supported."
            )

    def update(self, x, t, **model_kwargs):
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        x0 = self.diff_model(x, timestep, **model_kwargs)
        alpha = self.sde.alphas.to(x.device)[timestep, None, None, None]
        alphas_cumprod = self.sde.alphas_cumprod.to(x.device)[timestep, None, None, None]
        sqrt_alphas_cumprod = self.sde.sqrt_alphas_cumprod.to(x.device)[timestep, None, None, None]

        factor = (
            torch.sqrt(1. - alphas_cumprod)
            / torch.sqrt(1. - alphas_cumprod / alpha)
        )
        x_mean = (
            sqrt_alphas_cumprod * x0
            + factor * (x - torch.sqrt(alphas_cumprod / alpha) * x0)
        )
        x = x_mean
        return x, x_mean, x0


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""
    def __init__(self, **kwargs):
        pass

    def update(self, x, t, **model_kwargs):
        return x, x, None


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, diff_model, snr, n_steps, continuous):
        super().__init__(sde, diff_model, snr, n_steps)
        if not isinstance(sde, (VPSDE, VESDE)):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        self.score_fn = get_score_fn(sde, diff_model, continuous)

    def update(self, x, t, **model_kwargs):
        target_snr = self.snr
        if isinstance(self.sde, VPSDE):
            timestep = (t * (self.sde.N - 1) / self.sde.T).long()
            alpha = self.sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        x_mean = x
        for i in range(self.n_steps):
            grad = self.score_fn(x, t, **model_kwargs)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(
                grad.reshape(grad.shape[0], -1), dim=-1
            ).mean()
            noise_norm = torch.norm(
                noise.reshape(noise.shape[0], -1), dim=-1
            ).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean, grad


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""

    def __init__(self, sde, diff_model, snr, n_steps, continuous):
        super().__init__(sde, diff_model, snr, n_steps)
        if not isinstance(sde, (VPSDE, VESDE, subVPSDE)):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        self.score_fn = get_score_fn(sde, diff_model, continuous)

    def update(self, x, t, **model_kwargs):
        sde = self.sde
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, (VPSDE, subVPSDE)):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        x_mean = x
        for i in range(n_steps):
            grad = self.score_fn(x, t, **model_kwargs)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean, grad


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""
    def __init__(self, **kwargs):
        pass

    def update(self, x, t, **model_kwargs):
        return x, x, None

# -----------------------------------------------------------------------------


@register_sampler(name="pc")
class PCSampler():
    def __init__(
        self,
        *,
        predictor: str,
        corrector: str,
        model,
        shape,
        n_corrector_steps_each: int=1,
        snr: float=None,
        eps: float=1e-8,
        denoise: bool=True,
        **predictor_kwargs
    ):
        """Predictor-Corrector sampler.

        Args:
            predictor:
            corrector:
            model:
            shape:
            n_corrector_steps_each:
            snr:
            eps:
            denoise:
            **predictor_kwargs:
        """
        super().__init__()

        self.predictor = get_predictor(
            name=predictor,
            sde=model.sde,
            diff_model=model.diff_model,
            **predictor_kwargs
        )
        self.corrector = get_corrector(
            name=corrector,
            sde=model.sde,
            snr=snr,
            n_steps=n_corrector_steps_each,
            diff_model=model.diff_model,
        )

        self.model = model
        self.repr_ae = model.repr_ae
        self.sde = model.sde
        self.shape = shape
        self.eps = eps
        self.denoise = denoise
        self.device = model.device

    @torch.no_grad()
    def __call__(self, zT=None, stop_at_step_frac=None, **model_kwargs):
        """Computes a batch of samples.

        Args:
            **model_kwargs: Passed on to the model

        Returns: Samples
        """
        if zT is not None:
            x = zT
        else:
            x = self.sde.prior_sampling(self.shape).to(self.device)
        x_mean = None
        timesteps = torch.linspace(
            self.sde.T, self.eps, self.sde.N, device=self.device
        )
        if stop_at_step_frac is not None and stop_at_step_frac < 1.:
            timesteps = timesteps[:int(stop_at_step_frac*self.sde.N)]

        for t in timesteps:
            vec_t = torch.ones(self.shape[0], device=t.device) * t
            x, x_mean, _ = self.corrector.update(x, vec_t, **model_kwargs)
            x, x_mean, _ = self.predictor.update(x, vec_t, **model_kwargs)

        return x_mean if self.denoise else x

    @torch.no_grad()
    def sampling_progression(self, zT=None, **model_kwargs):
        """Sampling generator.

        Args:
            **model_kwargs: Passed on to the model

        Yields: Sampling step and Estimator ouput
        """
        if zT is not None:
            x = zT
        else:
            x = self.sde.prior_sampling(self.shape).to(self.device)
        x_mean = None
        timesteps = torch.linspace(
            self.sde.T, self.eps, self.sde.N, device=self.device
        )

        for t in timesteps:
            vec_t = torch.ones(self.shape[0], device=t.device) * t
            x, x_mean, estimate = self.corrector.update(
                x, vec_t, **model_kwargs
            )
            x, x_mean, estimate = self.predictor.update(
                x, vec_t, **model_kwargs
            )

            yield (x_mean if self.denoise else x, estimate)

        yield x_mean, None

    @torch.no_grad()
    def ddim_encode(
        self,
        input_batches,
        **model_kwargs
    ):
        x = input_batches
        timesteps = torch.linspace(
            self.eps, self.sde.T, self.sde.N, device=self.device
        )
        for t in timesteps[1:]:
            vec_t = torch.ones(input_batches.shape[0], device=t.device) * t
            x, _, _ = self.predictor.update(x, vec_t, **model_kwargs)
        return x


    @torch.no_grad()
    def reconstruct(
        self,
        input_batches,
        *,
        use_r_above=None,
        use_r_below=None,
        precomputed_r=None,
        sample_r=False,
        **model_kwargs
    ):
        """Reconstruct input batches via the representation model.

        Args:
            input_batches: Images to reconstruct
            use_r_above: Only use representation above given t-threshold
            use_r_below: Only use representation below given t-threshold
            **model_kwargs: Passed on to the model

        Returns: Reconstructions
        """
        assert self.repr_ae is not None, "No representation model available"
        assert not (use_r_below is not None and use_r_above is not None)

        x = self.sde.prior_sampling(input_batches.shape).to(self.device)
        x_mean = None
        timesteps = torch.linspace(
            self.sde.T, self.eps, self.sde.N, device=self.device
        )

        repr_enc_kwargs = {}

        if precomputed_r is None:
            if self.repr_ae.decoder.class_conditional:
                raise NotImplementedError()
            if self.repr_ae.encoder.class_conditional:
                assert "y" in model_kwargs
                repr_enc_kwargs["y"] = model_kwargs["y"]

            assert (
                self.repr_ae.encoder.time_conditional
                == self.repr_ae.decoder.time_conditional
            )

            if not self.repr_ae.decoder.time_conditional:
                if sample_r:
                    r = self.repr_ae.encode(
                        input_batches, **repr_enc_kwargs
                    ).sample()
                else:
                    r = self.repr_ae.encode(
                        input_batches, **repr_enc_kwargs
                    ).mode()

                model_kwargs["repr"] = self.repr_ae.decode(r)
        else:
            model_kwargs["repr"] = self.repr_ae.decode(precomputed_r)

        if use_r_above is not None:
            use_r = lambda t: t > use_r_above
        elif use_r_below is not None:
            use_r = lambda t: t < use_r_below
        else:
            use_r = lambda t: True

        for t in timesteps:
            vec_t = torch.ones(input_batches.shape[0], device=t.device) * t

            if self.repr_ae.decoder.time_conditional:

                if self.model.continuous:
                    time_cond = vec_t
                else:
                    time_cond = (vec_t*(self.sde.N-1) / self.sde.T).long()

                if use_r(t):
                    if sample_r:
                        r = self.repr_ae.encode(
                            input_batches, time_cond=time_cond
                        ).sample()
                    else:
                        r = self.repr_ae.encode(
                            input_batches, time_cond=time_cond
                        ).mode()
                else:
                    r = torch.randn(
                        size=(
                            (input_batches.shape[0],)
                            + self.repr_ae.latent_shape
                        ),
                        device=self.device,
                    )

                model_kwargs["repr"] = self.repr_ae.decode(
                    r, time_cond=time_cond
                )

            x, x_mean, _ = self.corrector.update(x, vec_t, **model_kwargs)
            x, x_mean, _ = self.predictor.update(x, vec_t, **model_kwargs)

        return x_mean if self.denoise else x

    @torch.no_grad()
    def reconstruction_progression(
        self,
        input_batches,
        *,
        precomputed_r=None,
        sample_r=False,
        use_r_above=None,
        use_r_below=None,
        **model_kwargs
    ):
        """Reconstruct input batches via the representation model.

        Args:
            input_batches: Images to reconstruct
            use_r_above: Only use representation above given t-threshold
            use_r_below: Only use representation below given t-threshold
            **model_kwargs: Passed on to the model

        Returns: Reconstructions
        """
        assert self.repr_ae is not None, "No representation model available"
        assert not (use_r_below is not None and use_r_above is not None)
        x = self.sde.prior_sampling(input_batches.shape).to(self.device)
        x_mean = None
        timesteps = torch.linspace(
            self.sde.T, self.eps, self.sde.N, device=self.device
        )

        repr_enc_kwargs = {}

        if precomputed_r is None:
            if self.repr_ae.decoder.class_conditional:
                raise NotImplementedError()
            if self.repr_ae.encoder.class_conditional:
                assert "y" in model_kwargs
                repr_enc_kwargs["y"] = model_kwargs["y"]

            assert (
                self.repr_ae.encoder.time_conditional
                == self.repr_ae.decoder.time_conditional
            )

            if not self.repr_ae.decoder.time_conditional:
                if sample_r:
                    r = self.repr_ae.encode(
                        input_batches, **repr_enc_kwargs
                    ).sample()
                else:
                    r = self.repr_ae.encode(
                        input_batches, **repr_enc_kwargs
                    ).mode()

                model_kwargs["repr"] = self.repr_ae.decode(r)
        else:
            model_kwargs["repr"] = self.repr_ae.decode(precomputed_r)

        if use_r_above is not None:
            use_r = lambda t: t > use_r_above
        elif use_r_below is not None:
            use_r = lambda t: t < use_r_below
        else:
            use_r = lambda t: True

        for t in timesteps:
            vec_t = torch.ones(input_batches.shape[0], device=t.device) * t

            if self.repr_ae.decoder.time_conditional:

                if self.model.continuous:
                    time_cond = vec_t
                else:
                    time_cond = (vec_t*(self.sde.N-1) / self.sde.T).long()

                if use_r(t):
                    r = self.repr_ae.encode(
                        input_batches, time_cond=time_cond
                    ).sample()
                else:
                    r = torch.randn(
                        size=(
                            (input_batches.shape[0],)
                            + self.repr_ae.latent_shape
                        ),
                        device=self.device,
                    )

                model_kwargs["repr"] = self.repr_ae.decode(
                    r, time_cond=time_cond
                )

            x, x_mean, estimate = self.corrector.update(
                x, vec_t, **model_kwargs
            )
            x, x_mean, estimate = self.predictor.update(
                x, vec_t, **model_kwargs
            )

            yield (x_mean if self.denoise else x, estimate)

        yield x_mean, None


@register_sampler(name="ode")
class ODESampler():
    """ODE sampler"""
    def __init__(
        self,
        *,
        sde,
        model,
        shape,
        denoise: bool=True,
        rtol: float=1e-5,
        atol: float=1e-5,
        method: str="RK45",
        eps: float=1e-3,
        device: str="cuda",
    ):
        """Probability flow ODE sampler with the black-box ODE solver.

        Args:
            sde: An `sde.SDE` object that represents the forward SDE.
            model: A score model.
            shape: The expected shape of a single sample.
            denoise: If True, add one-step denoising to final samples.
            rtol: The relative tolerance level of the ODE solver.
            atol: The absolute tolerance level of the ODE solver.
            method: The algorithm used for the black-box ODE solver.
                See the documentation of `scipy.integrate.solve_ivp`.
            eps: The reverse-time SDE/ODE will be integrated to `eps` for
                numerical stability.
            device: PyTorch device.
        """
        super().__init__()
        self.sde = sde
        self.model = model
        self.shape = shape
        self.denoise = denoise
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.eps = eps
        self.device = device

    def denoise_update_fn(self, x, **model_kwargs):
        score_fn = get_score_fn(self.sde, self.model, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(
            self.sde, score_fn, probability_flow=False
        )
        vec_eps = torch.ones(x.shape[0], device=x.device) * self.eps
        _, x = predictor_obj.update_fn(x, vec_eps, **model_kwargs)
        return x

    def drift_fn(self, x, t, **model_kwargs):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(self.sde, self.model, continuous=True)
        rsde = self.sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, **model_kwargs)[0]

    @torch.no_grad()
    def _sampling_progression(self, z=None, **model_kwargs):
        """Returns the whole integration solution"""
        if z is None:
            # If not represent, sample the latent code from the prior
            # distibution of the SDE.
            x = self.sde.prior_sampling(self.shape).to(self.device)
        else:
            x = z

        # Class conditioning
        if self.model.class_conditional:
            if "y" in model_kwargs:
                if type(model_kwargs["y"]) is int:
                    assert model_kwargs["y"] < self.model.num_classes
                    model_kwargs["y"] = torch.tensor(
                        self.shape[0] * [model_kwargs["y"]],
                        device=self.device,
                    )
            else:
                model_kwargs["y"] = torch.randint(
                    self.model.num_classes,
                    size=(self.shape[0],),
                    device=self.device
                )

        def ode_func(t, x):
            x = from_flattened_numpy(x, self.shape).to(self.device).type(
                torch.float32
            )
            vec_t = torch.ones(self.shape[0], device=x.device) * t
            drift = self.drift_fn(x, vec_t, **model_kwargs)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(
            ode_func,
            (self.sde.T, self.eps),
            to_flattened_numpy(x),
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
        )

    def __call__(self, z=None, **model_kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            z: If present, generate samples from latent code `z`.
            **model_kwargs: Passed on to the model

        Returns: Samples
        """
        solution = self._sampling_progression(z=z, **model_kwargs)
        # nfe = solution.nfev # number of function evaluations
        x = torch.tensor(
            solution.y[:, -1]
        ).reshape(self.shape).to(self.device).type(torch.float32)

        # Denoising is equivalent to running one predictor step without
        # adding noise
        if self.denoise:
            x = self.denoise_update_fn(x, **model_kwargs)

        return x

    def sampling_progression(self, z=None, **model_kwargs):
        """ODE sampling generator. Computes the whole sampling progression
        first, then yields the sub-steps.

        Args:
            z: If present, generate samples from latent code `z`.
            **model_kwargs: Passed on to the model

        Yields: Sampling step
        """
        samples = self._sampling_progression(z=z, **model_kwargs).y.T
        for s in samples:
            yield s
