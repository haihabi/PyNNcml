import torch
from torch import nn
from pynncml.datasets.link_data import handle_attenuation_input
from pynncml.datasets import MetaData
from pynncml.single_cml_methods.baseline import DynamicBaseLine
from pynncml.single_cml_methods.power_law import PowerLawType, PowerLaw


class OneStepDynamic(nn.Module):
    def __init__(self, power_law_type: PowerLawType, r_min: float, window_size: int, quantization_delta: float = 1.0):
        super(OneStepDynamic, self).__init__()
        self.bl = DynamicBaseLine(window_size)
        self.pl = PowerLaw(power_law_type, r_min)
        self.quantization_delta = quantization_delta

    def forward(self, data: torch.Tensor, metadata: MetaData) -> torch.Tensor:  # model forward pass
        att_max, att_min = handle_attenuation_input(data)
        bl_min = self.bl(att_min + self.quantization_delta / 2)
        att = att_max - bl_min - self.quantization_delta / 2
        rain_rate = self.pl(att, metadata.length, metadata.frequency, metadata.polarization)
        return rain_rate
