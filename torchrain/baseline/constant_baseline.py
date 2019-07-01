import torch
from torch import nn


class ConstantBaseLine(nn.Module):
    def __init__(self):
        super(ConstantBaseLine, self).__init__()

    def _single_link(self, attenuation: torch.Tensor, wd_classification: torch.Tensor):
        assert len(attenuation.shape) == 1
        baseline = [attenuation[0]]
        for i in range(1, attenuation.shape[0]):
            if wd_classification[i]:
                baseline.append(baseline[i - 1])
            else:
                baseline.append(attenuation[i])
        return torch.stack(baseline, dim=0)

    def forward(self, input_attenuation: torch.Tensor, input_wet_dry: torch.Tensor) -> torch.Tensor:
        """
        This is the forward function of Constant baseline model that implants the following equation:



        :param input_attenuation: A Tensor of shape $[N_b,N_s]$ where $N_b$ is the batch size and $N_s$ is the length of
        time sequence. This parameter is the attenuation tensor symbolized as $A_{i,n}$
        :param input_wet_dry: A Tensor of shape $[N_b,N_s]$ where $N_b$ is the batch size and $N_s$ is the length of
        time sequence. This parameter is the wet dry induction tensor symbolized as $\hat{y}^{wd}_{i,n}$
        :return: A Tensor of shape $[N_b,N_s]$ where $N_b$ is the batch size and $N_s$ is the length of
        time sequence. This parameter is the baseline tensor symbolized as $A^{\Delta}_{i,n}$
        """
        return torch.stack(
            [self._single_link(input_attenuation[batch_index, :], input_wet_dry[batch_index, :]) for batch_index in
             range(input_attenuation.shape[0])], dim=0)
