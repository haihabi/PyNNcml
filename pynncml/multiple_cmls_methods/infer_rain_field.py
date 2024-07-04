import torch
from torch import nn
from pynncml.multiple_cmls_methods.infer_multiple_cmls import InferMultipleCMLs


class InferRainField(nn.Module):
    def __init__(self, in_cml2rain_method, reconstruction_method):
        super().__init__()

        self.imc = InferMultipleCMLs(in_cml2rain_method)
        self.reconstruction_method = reconstruction_method

    def forward(self, link_set):
        rain_est = self.imc(link_set)
        rain_map = self.reconstruction_method(rain_est, link_set)
        return rain_map, rain_est

