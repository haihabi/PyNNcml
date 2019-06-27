import torch
import torchrain as tr
from torch import nn


class OneStepDynamic(nn.Module):
    def __init__(self, power_law_type: tr.power_law.PowerLawType, r_min: float, window_size: int):
        super(OneStepDynamic, self).__init__()
        self.bl = tr.baseline.ConstantBaseLine()
        self.pl = tr.power_law.PowerLaw(power_law_type, r_min)

    def forward(self, data: torch.Tensor, metadata: tr.MetaData) -> torch.Tensor:  # model forward pass
        wet_dry_classification = self.wd(data)
        bl_min = self.bl(data, wet_dry_classification)
        att = data - bl_min
        rain_rate = self.pl(att, metadata.length, metadata.frequency, metadata.polarization)
        return rain_rate
