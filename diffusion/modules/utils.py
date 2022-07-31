""""""
from typing import Union

import torch
from torch import nn
import numpy as np

from diffusion.modules.sde import VESDE, VPSDE, subVPSDE

# -----------------------------------------------------------------------------

class StepSamplingSchedule():
    """"""
    MODES = ["masked-down", "masked-up", "splitting", "none", "fixed"]

    def __init__(
        self,
        *,
        mode: str="none",
        continuous: bool,
        N: int=None,
        T: Union[float, int]=None,
        eps: float=1e-8,
        update_every_n: int=None,
        full_n_reached: int=None,
        fixed_value: Union[float, int]=None,
    ):
        """Scheduler for adaptive timestep sampling.

        Args:
            mode: Sampling Scheduling mode. May be one of:
                none: No schedule; sample uniformly from [0, T]
                masked-down: Sample from [l, T] where l decreases from T to 0
                masked-up: Sample from [0, u] where u increases from 0 to T
                splitting: Halves the timestep intervals with each update step
                    leading to the following sampling sequence:
                    {T}, {T/2, T}, {T/4, T/2, 3T/4, T}, ...
                fixed: Choose a constant value. Expects float in [0, T] for
                    continuous and int in [0, N-1] for non-continuous model.
            continuous: Whether to sample from continuous timestep interval.
                Otherwise, sample discrete timesteps.
            N: discrete timestep indices lie in [0, N-1]
            T: timesteps lie in [0, T]
            eps: epsilon for numerical stability
            update_every_n: Update schedule every n steps
            full_n_reached: Full N reached after the specified number of steps.
                Mutually exclusive with the ``update_every_n`` parameter.
            fixed_value: Fixed value for "fixed" mode
        """
        super().__init__()

        if mode.lower() not in self.MODES:
            raise ValueError(
                f"Invalid StepSamplingSchedule mode {mode}. Available modes: "
                f"{'.'.join(self.MODES)}"
            )

        if (
            mode.lower() not in ["none", "fixed"]
            and (update_every_n is None) == (full_n_reached is None)
        ):
            raise ValueError(
                "Please specify _one_ of: 'update_every_n', 'full_n_reached'"
            )

        self.mode = mode.lower() or "none"
        self.continuous = continuous
        self.N = N
        self.T = T
        self.eps = eps
        self.update_every_n = update_every_n
        self.full_n_reached = full_n_reached
        self.fixed_value = fixed_value

        self.n_split_steps = 0
        self.split_steps = [self.N - 1]
        self.split_range = self.T / self.N

    def get_all_steps(self, device=None):
        """"""
        if self.continuous:
            return torch.linspace(self.eps, self.T, self.N, device=device)
        else:
            return torch.linspace(0, self.N, self.N, device=device).long()

    def eval(self, step: int, *, num_samples, device=None):
        """"""
        n_full = None
        if self.mode not in ["none", "fixed"]:
            if self.full_n_reached is not None:
                n_full = self.full_n_reached
            else:
                n_full = self.N * self.update_every_n

        if self.mode == "none":
            if self.continuous:
                return (
                    torch.rand(num_samples, device=device)
                    * (self.T - self.eps) + self.eps
                )
            else:
                return torch.randint(
                    0, self.N, size=(num_samples,), device=device
                )

        elif self.mode == "fixed":
            if self.continuous:
                return torch.tensor(
                    [self.fixed_value] * num_samples, device=device
                )
            else:
                assert int(self.fixed_value) == self.fixed_value
                return torch.tensor(
                    [int(self.fixed_value)] * num_samples, device=device
                )

        elif self.mode == "masked-down":
            mask_below = max([1. - step / n_full - 1e-10, 0.])

            if self.continuous:
                return (
                    torch.rand(num_samples, device=device)
                    * (self.T * (1. - mask_below) - self.eps)
                    + mask_below * self.T + self.eps
                )
            else:
                mask_below = int(mask_below * self.N)
                return torch.randint(
                    mask_below, self.N, size=(num_samples,), device=device
                )

        elif self.mode == "masked-up":
            mask_above = min([step / n_full + 1e-10, 1.])

            if self.continuous:
                return (
                    torch.rand(num_samples, device=device)
                    * (self.T * mask_above - self.eps) + self.eps
                )
            else:
                mask_above = int(mask_above * self.N)
                return torch.randint(
                    0, mask_above, size=(num_samples,), device=device
                )

        elif self.mode == "splitting":
            if (
                self.n_split_steps >= self.N
                or 2 ** (self.n_split_steps + 1) >= self.N
            ):
                self.n_split_steps = self.N
                # sample from the whole interval
                if self.continuous:
                    return (
                        torch.rand(num_samples, device=device)
                        * (self.T - self.eps) + self.eps
                    )
                else:
                    return torch.randint(
                        0, self.N, size=(num_samples,), device=device
                    )

            else:
                split_steps_new = []
                for s in self.split_steps:
                    split_steps_new.append([s//2, s])

                self.split_steps = split_steps_new
                self.n_split_steps = int(2 ** (self.n_split_steps + 1))
                assert len(self.split_steps) == self.n_split_steps

                step_indices = torch.multinomial(
                    # sample from uniform distribution over split steps
                    torch.full((self.n_split_steps,), 1.),
                    num_samples=num_samples,
                    replacement=True,
                )
                step = torch.tensor(self.split_steps)[step_indices].to(device)

                if self.continuous:
                    # sample from range around selected split
                    shift_norm = torch.rand(num_samples, device=device)
                    # NOTE When selecting step T, positive shifts will be
                    #      clipped to T. This slight bias should be negligible.
                    return torch.clip(
                        step + (shift_norm * 2. - 1.) * (self.split_range),
                        min=0.,
                        max=self.T,
                    )
                else:
                    return step


class RunningStd(nn.Module):
    """Sequentially update a running sample standard deviation using a
    batch-wise version of Welford's online algorithm.
    """
    def __init__(self, device=None):
        super().__init__()

        # Counts number of seen batches
        self.register_buffer(
            "running_n_batches", torch.tensor(0, dtype=torch.int, device=device)
        )
        # Set batch-size with first batch and don't change afterwards
        self.register_buffer(
            "batch_size", torch.tensor(0, dtype=torch.int, device=device)
        )
        self.register_buffer(
            "running_mean", torch.tensor(0., dtype=torch.float32, device=device)
        )
        # M = sum([z - mean(z)]^2)
        self.register_buffer(
            "running_M", torch.tensor(0., dtype=torch.float32, device=device)
        )

    def forward(self, z):
        """Updates the running parameters given a new batch"""
        new_mean = (
            (self.running_n_batches * self.running_mean + z.mean())
            / (self.running_n_batches + 1)
        )

        if self.running_M == 0:
            self.running_M += torch.sum((z - new_mean)**2)
        else:
            self.running_M += torch.sum((z - new_mean) * (z - self.running_mean))

        self.running_mean.fill_(new_mean)
        self.running_n_batches += 1

        if self.batch_size == 0:
            self.batch_size += torch.numel(z)

    @property
    def std(self):
        """Returns the current std estimate"""
        return torch.sqrt(
            self.running_M / (self.running_n_batches * self.batch_size - 1)
        )

    # Alternative M-update
    # comparison = torch.sqrt(
    #         self.running_M / (self.running_n - 1)
    #         * (self.running_n - 1) / (self.running_n + num_el - 1)
    #         + z.std()**2 * (num_el - 1) / (self.running_n + num_el - 1)
    #         + self.running_n * num_el * (self.running_mean - z.mean())**2
    #         / (self.running_n + num_el) / (self.running_n + num_el - 1)
    # )


class DiagonalGaussianDistribution():
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 10.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[*range(1, self.mean.ndim)],
                )
            else:
                return 0.5 * torch.sum(
                    (
                        torch.pow(self.mean - other.mean, 2) / other.var
                        + self.var / other.var - 1.0 - self.logvar
                        + other.logvar
                    ),
                    dim=[*range(1, self.mean.ndim)],
                )

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            (
                logtwopi + self.logvar
                + torch.pow(sample - self.mean, 2) / self.var
            ),
            dim=[*range(1, self.mean.ndim)],
        )

    def mode(self):
        return self.mean


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened
    numpy array `x`.
    """
    return torch.from_numpy(x.reshape(shape))


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


def get_score_fn(sde, model, continuous: bool):
    """Prepare score function for continuous SDE-based models.

    Args:
        sde: An sde.SDE object that represents the forward SDE.
        model: A time-conditional diffusion model.
        continuous: If True, the diffusion model is expected to directly take
            continuous time steps.

    Returns:
        A score function.
    """
    if isinstance(sde, (VPSDE, subVPSDE)):
        def score_fn(x, t, y=None, **model_kwargs):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, subVPSDE):
                # For VP models, t=0 corresponds to the lowest noise level.
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                t_cond = (t * (sde.N - 1) / sde.T).long()
                score = model(x, t_cond, y=y, **model_kwargs)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]

            else:
                # For VP models, t=0 corresponds to the lowest noise level
                t_cond = (t * (sde.N - 1) / sde.T).long()
                score = model(x, t_cond, y=y, **model_kwargs)
                std = sde.sqrt_1m_alphas_cumprod.to(t_cond.device)[t_cond]

            score = - score / std[:, None, None, None]
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, t, y=None, **model_kwargs):
            if continuous:
                t_cond = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE models, t=0 corresponds to the highest noise level
                t_cond = sde.T - t
                t_cond *= sde.N - 1
                t_cond = torch.round(t_cond).long()

            score = model(x, t_cond, y=y, **model_kwargs)
            return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn
