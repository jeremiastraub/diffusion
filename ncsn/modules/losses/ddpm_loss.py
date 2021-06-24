""""""
import torch
import torch.nn as nn

from ..sde.sde import VPSDE

# -----------------------------------------------------------------------------


class DDPMLoss(nn.Module):
    """"""
    def __init__(self, reduce_mean: bool = True):
        super().__init__()
        self.reduce_mean = reduce_mean

    def forward(self, x, model, sde: VPSDE):
        """"""
        assert isinstance(sde, VPSDE), "DDPM training only works for VPSDEs."

        labels = torch.randint(0, sde.N, size=(x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        perturbed_data = (
            x * sde.sqrt_alphas_cumprod.to(x.device)[labels, None, None, None]
            + sde.sqrt_1m_alphas_cumprod.to(x.device)[labels, None, None, None]
            * noise
        )
        score = model(perturbed_data, labels)
        losses = torch.square(score - noise)

        if self.reduce_mean:
            loss = torch.mean(losses)
        else:
            loss = torch.mean(
                torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
            )
        return loss
