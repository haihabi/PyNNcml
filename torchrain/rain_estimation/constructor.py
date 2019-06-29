import torchrain as tr
from torchrain.rain_estimation.ts_constant import TwoStepConstant
from torchrain.rain_estimation.os_dynamic import OneStepDynamic


def two_step_constant_baseline(power_law_type: tr.power_law.PowerLawType, r_min: float, window_size: int,
                               threshold: float, wa_factor: float = None):
    if wa_factor is None:
        return TwoStepConstant(power_law_type, r_min, window_size, threshold)
    else:
        return TwoStepConstant(power_law_type, r_min, window_size, threshold, wa_factor=wa_factor)


def one_step_dynamic_baseline(power_law_type: tr.power_law.PowerLawType, r_min: float, window_size: int):
    return OneStepDynamic(power_law_type, r_min, window_size)


def two_step_network():
    raise NotImplemented


def one_step_network():
    raise NotImplemented
