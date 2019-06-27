import torch
import torchrain as tr
from torch import nn


class TwoStepConstant(nn.Module):
    def __init__(self, power_law_type: tr.power_law.PowerLawType, r_min: float, window_size: int, threshold: float):
        super(TwoStepConstant, self).__init__()
        self.wd = tr.wet_dry.STDWetDry(threshold, window_size)
        self.bl = tr.baseline.ConstantBaseLine()
        self.pl = tr.power_law.PowerLaw(power_law_type, r_min)

    def forward(self, data: torch.Tensor, metadata: tr.MetaData) -> (torch.Tensor, torch.Tensor):  # model forward pass
        wet_dry_classification = self.wd(data)
        bl_min = self.bl(data, wet_dry_classification)
        att = data - bl_min
        rain_rate = self.pl(att, metadata.length, metadata.frequency, metadata.polarization)
        return rain_rate, wet_dry_classification
