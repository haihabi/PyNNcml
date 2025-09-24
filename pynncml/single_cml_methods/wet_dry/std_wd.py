import torch
import numpy as np
from pynncml.datasets.alignment import handle_attenuation_input, AttenuationType
from torch import nn

from pynncml.neural_networks.base_neural_network import BaseCMLProcessingMethod


class STDWetDry(BaseCMLProcessingMethod):
    def __init__(self, th, n_steps):
        """
        This class create a wet-dry detection model based on the standard deviation of the CML data.

        :param th: floating point number that represent the threshold value.
        :param n_steps: integer that represent the step size.
        :param is_min_max: boolean that state if the threshold is minimum or maximum.

        return: None
        """
        super(STDWetDry, self).__init__(input_data_type=None, input_rate=None, output_rate=None)
        self.n_steps = n_steps
        self.th = th

    def forward(self, input_attenuation,input_meta_data=None):
        """
        This function calculate the wet-dry detection based on the standard deviation of the CML data.

        :param input_attenuation: tensor that represent the attenuation data.

        return: tensor
        """
        att_data = handle_attenuation_input(input_attenuation)
        if att_data.attenuation_type == AttenuationType.MinMax:
            att_max = att_data.attenuation_max
            att_min = att_data.attenuation_min
            if len(input_attenuation) == 3:
                input_attenuation = (att_max - att_min).reshape([input_attenuation.shape[0], -1])
            else:
                input_attenuation = (att_max - att_min).reshape([1, -1])
        else:
            input_attenuation = att_data.attenuation
        # model forward pass
        shift_begin = [input_attenuation.shape[0], (self.n_steps - 1) // 2]
        shift_end = [input_attenuation.shape[0], self.n_steps - 1 - shift_begin[1]]

        sigma_n_base = torch.stack(
            [torch.std(input_attenuation[:, np.maximum(0, i - self.n_steps + 1): (i + 1)], unbiased=False, dim=1) for i
             in
             range(self.n_steps - 1, input_attenuation.shape[1])], dim=1)

        sigma_n_base = torch.cat(
            [torch.zeros(shift_begin, device=input_attenuation.device), sigma_n_base,
             torch.zeros(shift_end, device=input_attenuation.device)], dim=1)
        sigma_n = sigma_n_base / (2 * self.th)
        res = torch.min(torch.round(sigma_n), torch.Tensor([1], device=input_attenuation.device))
        res = torch.max(res, torch.Tensor([0], device=input_attenuation.device))

        res = res - sigma_n
        return res.detach() + sigma_n, sigma_n_base
