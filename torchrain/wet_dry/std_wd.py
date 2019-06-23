import torch
import numpy as np
from torch import nn


class STDWetDry(nn.Module):
    def __init__(self, th, n_steps):
        super(STDWetDry, self).__init__()
        self.n_steps = n_steps
        self.th = th

    def forward(self, input_attenuation):  # model forward pass
        shift_begin = (self.n_steps - 1) // 2
        shift_end = self.n_steps - 1 - shift_begin

        sigma_n_base = torch.stack(
            [torch.std(input_attenuation[np.maximum(0, i - self.n_steps + 1): (i + 1)], unbiased=False) for i in
             range(self.n_steps - 1, len(input_attenuation))], dim=0)
        sigma_n_base = torch.cat([torch.zeros(shift_begin).cuda(), sigma_n_base, torch.zeros(shift_end).cuda()])

        sigma_n = sigma_n_base / (2 * self.th)
        res = torch.min(torch.round(sigma_n), torch.Tensor([1]).cuda())
        res = torch.max(res, torch.Tensor([0]).cuda())
        res = res - sigma_n
        return res.detach() + sigma_n
