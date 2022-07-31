""""""
import torch
import torch.nn as nn

from diffusion.modules.sde import VPSDE

# -----------------------------------------------------------------------------


class DDPMNoiseLoss(nn.Module):
    """"""
    def __init__(
        self,
        *,
        reduce_mean: bool = True,
        likelihood_weighting: bool = False,
        x0_weighting: bool = False,
    ):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.likelihood_weighting = likelihood_weighting
        self.x0_weighting = x0_weighting

    def forward(self, x, model, sde: VPSDE, t, **model_kwargs):
        """"""
        assert isinstance(sde, VPSDE), "DDPM training only works for VPSDEs."

        noise = torch.randn_like(x)
        perturbed_data = (
            x * sde.sqrt_alphas_cumprod.to(x.device)[t, None, None, None]
            + sde.sqrt_1m_alphas_cumprod.to(x.device)[t, None, None, None]
            * noise
        )

        if self.likelihood_weighting:
            weighting = (
                sde.discrete_betas[t, None, None, None].to(x.device)
                / (
                    2. * sde.alphas[t, None, None, None].to(x.device)
                    * (1.-sde.alphas_cumprod[t, None, None, None].to(x.device))
                )
            )
        elif self.x0_weighting:
            # equivalent to the x0-reweighting
            weighting = (
                (1. - sde.alphas_cumprod[t, None, None, None].to(x.device))
                / sde.alphas_cumprod[t, None, None, None].to(x.device)
            )
        else:
            weighting = 1.

        noise_pred = model(perturbed_data, t, **model_kwargs)
        losses = torch.square(noise_pred - noise) * weighting

        if self.reduce_mean:
            loss = torch.mean(losses)
        else:
            loss = torch.mean(
                torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
            )
        return loss
