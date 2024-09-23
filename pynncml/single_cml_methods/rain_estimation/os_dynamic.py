from typing import Tuple, Any

import torch
from torch import nn
from pynncml.datasets.link_data import handle_attenuation_input, AttenuationType
from pynncml.datasets import MetaData
from pynncml.single_cml_methods.baseline import DynamicBaseLine
from pynncml.single_cml_methods.power_law import PowerLawType, PowerLaw


class OneStepDynamic(nn.Module):
    def __init__(self, power_law_type: PowerLawType, r_min: float, window_size: int, quantization_delta: float = 1.0):
        """
        This function create a one step dynamic baseline model. The model is used to estimate the rain rate from the CML data.
        This function also includes the quantization delta parameter for bias correction.
        :param power_law_type: enum that define the type of the power law.
        :param r_min: floating point number that represent the minimum value of the rain rate.
        :param window_size: integer that represent the window size.
        :param quantization_delta: floating point number that represent the quantization delta
        """
        super(OneStepDynamic, self).__init__()
        self.bl = DynamicBaseLine(window_size)
        self.pl = PowerLaw(power_law_type, r_min)
        self.quantization_delta = quantization_delta

    def forward(self, data: torch.Tensor, metadata: MetaData) -> tuple[Any, Any]:
        """
        This is the module forward function
        :param data: A tensor of attenuation.
        :param metadata: A metadata class that hold the metadata of the CML data.
        :return: A tuple of rain rate and baseline.
        """
        att_data = handle_attenuation_input(data)
        if att_data.attenuation_type == AttenuationType.MIN_MAX:
            att_max = att_data.attenuation_max
            att_min = att_data.attenuation_min
        else:
            att_min = att_data.attenuation
            att_max = att_data.attenuation
        bl_min = self.bl(att_min + self.quantization_delta / 2)
        att = att_max - bl_min - self.quantization_delta / 2
        rain_rate = self.pl(att, metadata.length, metadata.frequency, metadata.polarization)
        return rain_rate, bl_min
