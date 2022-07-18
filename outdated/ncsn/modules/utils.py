""""""

import torch

from .sde import VESDE, VPSDE, subVPSDE

# -----------------------------------------------------------------------------

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened
    numpy array `x`.
    """
    return torch.from_numpy(x.reshape(shape))


def get_score_fn(sde, model, continuous: bool):
    """Wraps `score_fn` so that the model output corresponds to a real
    time-dependent score function.

    Args:
        sde: An sde.SDE object that represents the forward SDE.
        model: A score model.
        continuous: If True, the score-based model is expected to directly take
            continuous time steps.

    Returns:
        A score function.
    """
    if isinstance(sde, (VPSDE, subVPSDE)):
        def score_fn(x, t, y=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, subVPSDE):
                # For VP models, t=0 corresponds to the lowest noise level.
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                t_cond = t * 999
                score = model(x, t_cond, y=y)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP models, t=0 corresponds to the lowest noise level
                t_cond = t * (sde.N - 1)
                score = model(x, t_cond, y=y)
                std = sde.sqrt_1m_alphas_cumprod.to(
                    t_cond.device
                )[t_cond.long()]

            score = - score / std[:, None, None, None]
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, t, y=None):
            if continuous:
                t_cond = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE models, t=0 corresponds to the highest noise level
                t_cond = sde.T - t
                t_cond *= sde.N - 1
                t_cond = torch.round(t_cond).long()

            score = model(x, t_cond, y=y)
            return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn
