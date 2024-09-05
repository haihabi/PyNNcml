import torch
from torch import nn
from pynncml.single_cml_methods.power_law import PowerLaw, PowerLawType
from pynncml.single_cml_methods.wet_dry import STDWetDry
from pynncml.single_cml_methods.baseline import ConstantBaseLine
from pynncml.datasets.link_data import handle_attenuation_input
from pynncml.datasets import MetaData


class TwoStepConstant(nn.Module):
    def __init__(self, power_law_type: PowerLawType, r_min: float, window_size: int, threshold: float,
                 wa_factor: float = 0.0):
        super(TwoStepConstant, self).__init__()
        self.wd = STDWetDry(threshold, window_size, is_min_max=PowerLawType.MAX == power_law_type)
        self.bl = ConstantBaseLine()
        self.wa_factor = wa_factor
        self.pl = PowerLaw(power_law_type, r_min)

    def forward(self, data: torch.Tensor, metadata: MetaData) -> (torch.Tensor, torch.Tensor):  # model forward pass
        att_max, att_min = handle_attenuation_input(data)

        wet_dry_classification, _ = self.wd(data)
        bl_min = self.bl(att_min, wet_dry_classification)
        att = att_max - bl_min - self.wa_factor
        rain_rate = self.pl(att, metadata.length, metadata.frequency, metadata.polarization)
        return rain_rate, wet_dry_classification
