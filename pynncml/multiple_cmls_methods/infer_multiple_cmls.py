import torch
from torch import nn


class InferMultipleCMLs(nn.Module):
    def __init__(self, in_cml2rain_method):
        super().__init__()
        self.cml2rain = in_cml2rain_method

    def forward(self, link_set):
        res_list = []
        for link in link_set:
            rain_est = self.cml2rain(link.attenuation(), link.meta_data)
            res_list.append(rain_est.flatten())
        return torch.stack(res_list, dim=0)
