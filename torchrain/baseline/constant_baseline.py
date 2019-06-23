import torch
from torch import nn


class ConstantBaseLine(nn.Module):
    def __init__(self):
        super(ConstantBaseLine, self).__init__()

    def forward(self, input_attenuation, input_wet_dry):  # model forward pass
        baseline = [input_attenuation[0]]
        for i in range(1, len(input_attenuation)):
            if input_wet_dry[i]:
                baseline.append(baseline[i - 1])
            else:
                baseline.append(input_attenuation[i])
        return torch.stack(baseline, dim=0)
