
import torch
from torch import nn
from pynncml.multiple_cmls_methods import InferMultipleCMLs


class XarrayInferenceEngine(nn.Module):
    def __init__(self,in_cml2rain_method, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_engine = InferMultipleCMLs(in_cml2rain_method)


    def forward(self, x_xarray):
        pass