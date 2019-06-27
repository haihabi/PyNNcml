import torchrain as tr
from torchrain.rain_estimation.ts_constant import TwoStepConstant


def two_step_constant_baseline(power_law_type: tr.power_law.PowerLawType, r_min: float, window_size: int,
                               threshold: float):
    return TwoStepConstant(power_law_type, r_min, window_size, threshold)
