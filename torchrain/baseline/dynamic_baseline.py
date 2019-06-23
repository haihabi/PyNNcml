import torch
import numpy as np
from torch import nn


class DynamicBaseLine(nn.Module):
    def __init__(self, n_steps):
        super(DynamicBaseLine, self).__init__()
        self.n_steps = n_steps

    def forward(self, input_attenuation):  # model forward pass
        return torch.stack([torch.min(input_attenuation[np.maximum(0, i - self.n_steps + 1): (i + 1)]) for i in
                            range(len(input_attenuation))], dim=0)
