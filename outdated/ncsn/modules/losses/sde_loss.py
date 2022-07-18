""""""

import torch
import torch.nn as nn

from ..sde import VPSDE, VESDE, subVPSDE, SDE

# -----------------------------------------------------------------------------


class SDELoss(nn.Module):
    """"""
    def __init__(
        self,
        *,
        likelihood_weighting: bool = False,
        reduce_mean: bool = True,
        eps: float = 1e-5,
    ):
        """Initializes a SDELoss instance.

        Args:
            likelihood_weighting (bool): Whether to weight the mixture of score
                matching losses according to https://arxiv.org/abs/2101.09258.
                If False, use weighting as recommended in Song et al., 2020.
            reduce_mean (bool): If True, average the loss across data
                dimensions. If False, sum the loss across data dimensions.
            eps (float): The smallest time step to sample from
        """
        super().__init__()
        self.eps = eps
        self.likelihood_weighting = likelihood_weighting

        # TODO In the original code, they use 0.5 * sum(...); why??

        self.reduce_op = (
            torch.mean if reduce_mean
            else lambda *args, **kwargs: torch.sum(*args, **kwargs)
        )

    def forward(self, x, model, sde: SDE, y=None):
        """Evaluate the loss.

        Args:
            x: mini-batch input data
            model: SDE model instance
            sde: SDE instance
            y: (class) conditioning

        Returns: loss
        """
        t = (
            torch.rand(x.shape[0], device=x.device) * (sde.T - self.eps)
            + self.eps
        )
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_data = mean + std[:, None, None, None] * z

        # Evaluate scores depending on the SDE type
        if isinstance(sde, (VPSDE, subVPSDE)):
            # Scale neural network output by standard deviation and flip sign.
            # For VP-trained models, t=0 corresponds to the lowest noise level.
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            t_cond = t * 999
            score = (
                - model(perturbed_data, t_cond, y=y)
                / std[:, None, None, None]
            )

        elif isinstance(sde, VESDE):
            # For VE-trained models, t=0 corresponds to the highest noise level
            score = model(perturbed_data, std, y=y)

        else:
            raise ValueError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

        if self.likelihood_weighting:
            g2 = sde.sde(torch.zeros_like(x), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = self.reduce_op(
                losses.reshape(losses.shape[0], -1), dim=-1
            ) * g2

        else:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = self.reduce_op(
                losses.reshape(losses.shape[0], -1), dim=-1
            )

        loss = torch.mean(losses)
        return loss
