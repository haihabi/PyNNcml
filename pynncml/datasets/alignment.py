import torch
from dataclasses import dataclass
from enum import Enum


class AttenuationType(Enum):
    """
    Attenuation type enumeration

    """
    MinMax = 'min_max'
    Instance = 'regular'

def create_data_alignment(in_data_type: AttenuationType, output_data_type:AttenuationType,
                          input_rate:int,output_rate:int):
    if output_rate<input_rate:
        raise ValueError("output rate should be greater than input rate")
    if in_data_type == AttenuationType.MinMax and output_data_type==AttenuationType.Instance:
        raise ValueError("Input MinMax data type is not supported for output Instance data type")
    pass


@dataclass
class AttenuationData:
    """
    Attenuation data class
    :param attenuation_min: torch.Tensor
    :param attenuation_max: torch.Tensor
    :param attenuation: torch.Tensor
    :param attenuation_type: AttenuationType
    """
    attenuation_min: torch.Tensor
    attenuation_max: torch.Tensor
    attenuation: torch.Tensor
    attenuation_type: AttenuationType


def handle_attenuation_input(attenuation: torch.Tensor) -> AttenuationData:
    """
    Handle the attenuation input and return the attenuation data
    :param attenuation: torch.Tensor
    :return: AttenuationData
    """
    attenuation_avg = att_min = att_max = None
    if len(attenuation.shape) == 2:
        attenuation_avg = attenuation
        attenuation_type = AttenuationType.Instance
    elif len(attenuation.shape) == 3 and attenuation.shape[2] == 2:
        att_max, att_min = attenuation[:, :, 0], attenuation[:, :, 1]  # split the attenuation to max and min
        attenuation_type = AttenuationType.MinMax
    else:
        raise Exception('The input attenuation vector dont match min max format or regular format')
    return AttenuationData(attenuation_min=att_min,
                           attenuation_max=att_max,
                           attenuation=attenuation_avg,
                           attenuation_type=attenuation_type)



