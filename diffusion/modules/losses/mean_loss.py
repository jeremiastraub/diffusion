""""""
import torch
import torch.nn as nn

from diffusion.modules.sde import VPSDE

# -----------------------------------------------------------------------------


class DDPMMeanLoss(nn.Module):
    """"""
    def __init__(
        self,
        *,
        reduce_mean: bool = True,
        likelihood_weighting: bool = False,
        eps_weighting: bool = False,
    ):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.likelihood_weighting = likelihood_weighting
        self.eps_weighting = eps_weighting

    def forward(self, x, model, sde: VPSDE, t, **model_kwargs):
        """"""
        assert isinstance(sde, VPSDE), "DDPM training only works for VPSDEs."

        noise = torch.randn_like(x)
        perturbed_data = (
            x * sde.sqrt_alphas_cumprod.to(x.device)[t, None, None, None]
            + sde.sqrt_1m_alphas_cumprod.to(x.device)[t, None, None, None]
            * noise
        )

        mu = (
            perturbed_data - (
                sde.discrete_betas.to(x.device)[t, None, None, None] * noise
                / sde.sqrt_1m_alphas_cumprod.to(x.device)[t, None, None, None]
             )
        ) / torch.sqrt(sde.alphas.to(x.device)[t, None, None, None])

        mu_pred = model(perturbed_data, t, **model_kwargs)

        if self.likelihood_weighting:
            weighting = (
                1. / (2.*sde.discrete_betas.to(x.device)[t, None, None, None])
            )
        elif self.eps_weighting:
            weighting = (
                sde.alphas.to(x.device)[t, None, None, None]
                * (1. - sde.alphas_cumprod.to(x.device)[t, None, None, None])
                / sde.discrete_betas.to(x.device)[t, None, None, None] ** 2.
            )
        else:
            weighting = 1. # drop all prefactors

        losses = torch.square(mu_pred - mu) * weighting

        if self.reduce_mean:
            loss = torch.mean(losses)
        else:
            loss = torch.mean(
                torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
            )
        return loss
