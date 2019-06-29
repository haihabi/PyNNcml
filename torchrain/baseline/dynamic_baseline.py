import torch
import numpy as np
from torch import nn


class DynamicBaseLine(nn.Module):
    def __init__(self, n_steps):
        super(DynamicBaseLine, self).__init__()
        self.n_steps = n_steps

    def forward(self, input_attenuation: torch.Tensor) -> torch.Tensor:  # model forward pass
        if len(input_attenuation.shape) != 2:
            raise Exception('Dynamic base line module only accepts inputs with shape length equal to 2')
        return torch.stack([self._single_link(input_attenuation[i, :]) for i in range(input_attenuation.shape[0])],
                           dim=0)

    def _single_link(self, input_attenuation: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.min(input_attenuation[np.maximum(0, i - self.n_steps + 1): (i + 1)]) for i in
                            range(input_attenuation.shape[0])], dim=0)
