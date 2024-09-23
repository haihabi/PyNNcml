import torch
from torch import nn
from pynncml.single_cml_methods.power_law import PowerLaw, PowerLawType
from pynncml.single_cml_methods.wet_dry import STDWetDry
from pynncml.single_cml_methods.baseline import ConstantBaseLine
from pynncml.datasets.link_data import handle_attenuation_input, AttenuationType
from pynncml.datasets import MetaData


class TwoStepsConstant(nn.Module):
    def __init__(self, power_law_type: PowerLawType, r_min: float, window_size: int, threshold: float,
                 wa_factor: float = 0.0):
        """
        This function create a two steps constant baseline model. The model is used to estimate the rain rate from the CML data.
        :param power_law_type: enum that define the type of the power law.
        :param r_min: floating point number that represent the minimum value of the rain rate.
        :param window_size: integer that represent the window size.
        :param threshold: floating point number that represent the threshold value.
        :param wa_factor: floating point number that represent the weight average factor.
        """
        super(TwoStepsConstant, self).__init__()
        self.wd = STDWetDry(threshold, window_size, is_min_max=PowerLawType.MAX == power_law_type)
        self.bl = ConstantBaseLine()
        self.wa_factor = wa_factor
        self.pl = PowerLaw(power_law_type, r_min)

    def forward(self, data: torch.Tensor, metadata: MetaData) -> (
            torch.Tensor, torch.Tensor, torch.Tensor):  # model forward pass
        """
        This is the module forward function
        :param data: A tensor of attenuation.
        :param metadata: A metadata class that hold the metadata of the CML data.
        """
        att_data = handle_attenuation_input(data)
        if att_data.attenuation_type == AttenuationType.MIN_MAX:
            att_max = att_data.attenuation_max
            att_min = att_data.attenuation_min
        elif att_data.attenuation_type == AttenuationType.REGULAR:
            att_min = att_data.attenuation
            att_max = att_data.attenuation
        else:
            raise ValueError("Attenuation type not supported")

        wet_dry_classification, _ = self.wd(data)
        bl_min = self.bl(att_min, wet_dry_classification)
        att = att_max - bl_min - self.wa_factor
        rain_rate = self.pl(att, metadata.length, metadata.frequency, metadata.polarization)
        return rain_rate, wet_dry_classification, bl_min
