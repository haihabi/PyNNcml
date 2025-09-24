
from torch import nn
from pynncml.datasets.xarray_processing import xarray2link
from pynncml.multiple_cmls_methods import InferMultipleCMLs


class XarrayInferenceEngine(nn.Module):
    def __init__(self,in_cml2rain_method,is_recurrent=True,is_attenuation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_engine = InferMultipleCMLs(in_cml2rain_method,is_recurrent,is_attenuation)


    def forward(self, x_xarray):
        link_set=xarray2link(x_xarray)
        return self.inference_engine(link_set)