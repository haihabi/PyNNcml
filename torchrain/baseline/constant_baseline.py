import torch
from torch import nn


class ConstantBaseLine(nn.Module):
    def __init__(self):
        super(ConstantBaseLine, self).__init__()

    def _single_link(self, attenuation, wd_classification):
        baseline = [attenuation[0]]
        for i in range(1, len(attenuation)):
            if wd_classification[i]:
                baseline.append(baseline[i - 1])
            else:
                baseline.append(attenuation[i])
        return torch.stack(baseline, dim=0)

    def forward(self, input_attenuation, input_wet_dry):  # model forward pass\
        return torch.stack(
            [self._single_link(input_attenuation[batch_index, :], input_wet_dry[batch_index, :]) for batch_index in
             range(input_attenuation.shape[0])], dim=0)
