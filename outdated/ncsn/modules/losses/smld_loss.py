""""""
import torch
import torch.nn as nn

from ..sde.sde import VESDE

# -----------------------------------------------------------------------------


class SMLDLoss(nn.Module):
    """"""
    def __init__(self, reduce_mean: bool = True):
        super().__init__()
        self.reduce_mean = reduce_mean

    def forward(self, x, model, sde: VESDE):
        """"""
        assert isinstance(sde, VESDE), "SMLD training only works for VESDEs."

        labels = torch.randint(0, sde.N, size=(x.shape[0],), device=x.device)
        # Previous SMLD models assume descending sigmas
        smld_sigma_array = torch.flip(sde.discrete_sigmas, dims=(0,))
        sigmas = smld_sigma_array.to(x.device)[labels]
        noise = torch.randn_like(x) * sigmas[:, None, None, None]
        perturbed_data = x + noise
        score = model(perturbed_data, labels)
        target = - noise / (sigmas**2)[:, None, None, None]
        losses = torch.square(score - target)

        if self.reduce_mean:
            loss = torch.mean(losses)
        else:
            loss = torch.mean(
                torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
            )
        return loss
