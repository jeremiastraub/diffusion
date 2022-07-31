""""""

import torch
import torch.nn as nn

from diffusion.modules.sde import VPSDE, VESDE, subVPSDE, SDE

# -----------------------------------------------------------------------------


class SDELoss(nn.Module):
    """"""
    def __init__(
        self,
        *,
        likelihood_weighting: bool = False,
        reduce_mean: bool = True,
    ):
        """SDE loss for continuous timesteps.

        Args:
            likelihood_weighting (bool): Whether to weight the mixture of score
                matching losses according to https://arxiv.org/abs/2101.09258.
                If False, use weighting as recommended in Song et al., 2020.
            reduce_mean (bool): If True, average the loss across data
                dimensions. If False, sum the loss across data dimensions.
        """
        super().__init__()
        self.likelihood_weighting = likelihood_weighting

        # TODO In the original code, they use 0.5 * sum(...); why??

        self.reduce_op = (
            torch.mean if reduce_mean
            else lambda *args, **kwargs: torch.sum(*args, **kwargs)
        )

    def forward(self, x, model, sde: SDE, t, **model_kwargs):
        """Evaluate the loss.

        Args:
            x: mini-batch input data
            model: SDE model instance
            sde: SDE instance
            t: time conditioning
            **model_kwargs: Passed to the score model forward method

        Returns: loss
        """
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_data = mean + std[:, None, None, None] * z

        # Evaluate scores depending on the SDE type
        if isinstance(sde, (VPSDE, subVPSDE)):
            # Scale neural network output by standard deviation and flip sign.
            # For VP-trained models, t=0 corresponds to the lowest noise level.
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.

            # TODO
            # - dividing by std is not the same as the dividing by sigma as it
            #   is done in the NCSN. check wether computations are correct
            # - keep difference between DDPMLoss and SDELoss with VPSDE. The
            #   former is the classical DDPMLoss, the latter is the equivalent
            #   but calculated from the continuous SDE-formulation
            #   (alpha_cumprod vs marginal_prob)
            # - The hard-coded 999 should be (sde.N - 1), right?

            t_cond = t * (sde.N - 1) # 999
            score = (
                - model(perturbed_data, t_cond, **model_kwargs)
                / std[:, None, None, None]
            )

        elif isinstance(sde, VESDE):
            # For VE-trained models, t=0 corresponds to the highest noise level
            score = model(perturbed_data, std, **model_kwargs)

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
