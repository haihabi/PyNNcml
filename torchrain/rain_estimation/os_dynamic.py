import torch
import torchrain as tr
from torch import nn
from torchrain.data_common import handle_attenuation_input


class OneStepDynamic(nn.Module):
    def __init__(self, power_law_type: tr.power_law.PowerLawType, r_min: float, window_size: int):
        super(OneStepDynamic, self).__init__()
        self.bl = tr.baseline.DynamicBaseLine(window_size)
        self.pl = tr.power_law.PowerLaw(power_law_type, r_min)

    def forward(self, data: torch.Tensor, metadata: tr.MetaData) -> torch.Tensor:  # model forward pass
        att_max, att_min = handle_attenuation_input(data)
        bl_min = self.bl(att_min + 0.5)
        att = att_max - bl_min - 0.5
        rain_rate = self.pl(att, metadata.length, metadata.frequency, metadata.polarization)
        return rain_rate
