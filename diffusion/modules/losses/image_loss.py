""""""
import torch
import torch.nn as nn

from diffusion.modules.sde import VPSDE

# -----------------------------------------------------------------------------


class DDPMImageLoss(nn.Module):
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

    def forward(self, x, model, sde: VPSDE, t, warped_x=None, **model_kwargs):
        """"""
        assert isinstance(sde, VPSDE), "DDPM training only works for VPSDEs."

        noise = torch.randn_like(x)

        if warped_x is not None:
            perturbed_data = (
                warped_x * sde.sqrt_alphas_cumprod.to(x.device)[t, None, None, None]
                + sde.sqrt_1m_alphas_cumprod.to(x.device)[t, None, None, None]
                * noise
            )
        else:
            perturbed_data = (
                x * sde.sqrt_alphas_cumprod.to(x.device)[t, None, None, None]
                + sde.sqrt_1m_alphas_cumprod.to(x.device)[t, None, None, None]
                * noise
            )

        x0_pred = model(perturbed_data, t, **model_kwargs)

        if self.likelihood_weighting:
            weighting = (
                sde.alphas_cumprod[t, None, None, None].to(x.device)
                * sde.discrete_betas[t, None, None, None].to(x.device)
                / (
                    2. * sde.alphas[t, None, None, None].to(x.device)
                    * (1. - sde.alphas_cumprod[t, None, None, None].to(x.device)) ** 2
                )
            )
        elif self.eps_weighting:
            # equivalent to the epsilon-reweighting
            weighting = (
                sde.alphas_cumprod[t, None, None, None].to(x.device)
                / (1. - sde.alphas_cumprod[t, None, None, None].to(x.device))
            )
        else:
            weighting = 1. # drop all prefactors

        losses = torch.square(x0_pred - x) * weighting

        if self.reduce_mean:
            loss = torch.mean(losses)
        else:
            loss = torch.mean(
                torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
            )
        return loss
