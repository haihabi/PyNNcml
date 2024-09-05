import torch
import numpy as np
from pynncml.datasets.link_data import handle_attenuation_input
from torch import nn


class STDWetDry(nn.Module):
    def __init__(self, th, n_steps, is_min_max=False):
        super(STDWetDry, self).__init__()
        self.n_steps = n_steps
        self.th = th
        self.is_min_max = is_min_max

    def forward(self, input_attenuation):
        if self.is_min_max:
            att_max, att_min = handle_attenuation_input(input_attenuation)
            if len(input_attenuation) == 3:
                input_attenuation = (att_max - att_min).reshape([input_attenuation.shape[0], -1])
            else:
                input_attenuation = (att_max - att_min).reshape([1, -1])
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
