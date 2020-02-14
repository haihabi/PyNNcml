import torch
import pynncml as pnc
from torch import nn
from pynncml.data_common import handle_attenuation_input


class TwoStepConstant(nn.Module):
    def __init__(self, power_law_type: pnc.power_law.PowerLawType, r_min: float, window_size: int, threshold: float,
                 wa_factor: float = 0.0):
        super(TwoStepConstant, self).__init__()
        self.wd = pnc.wet_dry.STDWetDry(threshold, window_size)
        self.bl = pnc.baseline.ConstantBaseLine()
        self.wa_factor = wa_factor
        self.pl = pnc.power_law.PowerLaw(power_law_type, r_min)

    def forward(self, data: torch.Tensor, metadata: pnc.MetaData) -> (torch.Tensor, torch.Tensor):  # model forward pass
        att_max, att_min = handle_attenuation_input(data)

        wet_dry_classification, _ = self.wd(att_max)
        bl_min = self.bl(att_min, wet_dry_classification)
        att = att_max - bl_min - self.wa_factor
        rain_rate = self.pl(att, metadata.length, metadata.frequency, metadata.polarization)
        return rain_rate, wet_dry_classification
