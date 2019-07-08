from torch import nn
import torch
import numpy as np
from enum import Enum
from scipy.interpolate import interp1d

ITU_TABLE = np.array([
    [1.000e+0, 2.000e+0, 4.000e+0, 6.000e+0, 7.000e+0, 8.000e+0, 1.000e+1,
     1.200e+1, 1.500e+1, 2.000e+1, 2.500e+1, 3.000e+1, 3.500e+1, 4.000e+1,
     4.500e+1, 5.000e+1, 6.000e+1, 7.000e+1, 8.000e+1, 9.000e+1, 1.000e+2],
    [3.870e-5, 2.000e-4, 6.000e-4, 1.800e-3, 3.000e-3, 4.500e-3, 1.010e-2,
     1.880e-2, 3.670e-2, 7.510e-2, 1.240e-1, 1.870e-1, 2.630e-1, 3.500e-1,
     4.420e-1, 5.360e-1, 7.070e-1, 8.510e-1, 9.750e-1, 1.060e+0, 1.120e+0],
    [3.520e-5, 1.000e-4, 6.000e-4, 1.600e-3, 2.600e-3, 4.000e-3, 8.900e-3,
     1.680e-2, 3.350e-2, 6.910e-2, 1.130e-1, 1.670e-1, 2.330e-1, 3.100e-1,
     3.930e-1, 4.790e-1, 6.420e-1, 7.840e-1, 9.060e-1, 9.990e-1, 1.060e+0],
    [9.120e-1, 9.630e-1, 1.121e+0, 1.308e+0, 1.332e+0, 1.327e+0, 1.276e+0,
     1.217e+0, 1.154e+0, 1.099e+0, 1.061e+0, 1.021e+0, 9.790e-1, 9.390e-1,
     9.030e-1, 8.730e-1, 8.260e-1, 7.930e-1, 7.690e-1, 7.530e-1, 7.430e-1],
    [8.800e-1, 9.230e-1, 1.075e+0, 1.265e+0, 1.312e+0, 1.310e+0, 1.264e+0,
     1.200e+0, 1.128e+0, 1.065e+0, 1.030e+0, 1.000e+0, 9.630e-1, 9.290e-1,
     8.970e-1, 8.680e-1, 8.240e-1, 7.930e-1, 7.690e-1, 7.540e-1, 7.440e-1]])
EULER_GAMMA = 0.57721566
FREQMAX = 100
FREQMIN = 1


class PowerLawType(Enum):
    r"""
    Power Law Type select between  max attenuation and instance attenuation
    """
    INSTANCE = 0
    MAX = 1


class PowerLaw(nn.Module):
    r"""
    The PowerLaw Module is implanted two type attenuation: max and instance attenuation as define in the following equations:
        The instance power law:
            .. math::
                R_n=\Big(\frac{1}{aL}\Big)^{\frac{1}{b}}A_n^{\frac{1}{b}}
        The max power law:
            .. math::
                R_n=\Big(\frac{1}{a(log(k)+\gamma)L}\Big)^{\frac{1}{b}}A_n^{\frac{1}{b}} \\
        where

    :param input_type: an Enum that config the current type of input attenuation
    :param r_min: a float number setting the minimal amount of rain.
    :param k: an integer value

    """

    def __init__(self, input_type: PowerLawType, r_min: float, k: int = 90):
        super(PowerLaw, self).__init__()
        self.r_min = r_min
        self.k = k
        self.input_type = input_type

    def forward(self, input_attenuation: torch.Tensor, length: float, frequency: float,
                polarization: bool) -> torch.Tensor:  # model forward pass
        """
        This is the module forward function

        :param input_attenuation: A tensor of attenuation of any shape.
        :param length:
        :param frequency:
        :param polarization:
        :return: A tensor of rain with the same shape as the input_attenuation tensor.
        """
        a, b = a_b_parameters(frequency, polarization)
        if self.input_type == PowerLawType.MAX:
            a = a * (np.log(self.k) + EULER_GAMMA) ** b
        beta = 1 / b
        alpha = np.power(1 / (a * length), beta)
        att = input_attenuation * (input_attenuation > 0).float()
        rain_rate = alpha * torch.pow(att, beta)
        rain_rate[rain_rate < self.r_min] = 0  # zero rain value below minmal rain
        return rain_rate


def a_b_parameters(frequency: float, polarization: bool) -> (float, float):
    """This function return the Power Law parameters which approximate the relation between A-R
       as define in [1].

    :param frequency: a floating point number represent the frequency in GHz (value must be between 1 Ghz and 100 GHz)
    :param polarization: boolean flag represent the polarization True - vertical and False -  horizontal
    :returns: a,b Power Law parameters

    References:
    [1] ITU, "ITU-R: Specific attenuation model for rain for use in prediction methods", International Telecommunication Union, 2013
    """
    frequncey = np.asarray(frequency)

    if frequncey.min() < FREQMIN or frequncey.max() > FREQMAX:
        raise ValueError('Frequency must be between {} Ghz and {} GHz.'.format(FREQMIN, FREQMAX))

    if polarization == 1:  # V
        f_a = interp1d(ITU_TABLE[0, :], ITU_TABLE[2, :], kind='cubic')
        f_b = interp1d(ITU_TABLE[0, :], ITU_TABLE[4, :], kind='cubic')
    elif polarization == 0:  # H
        f_a = interp1d(ITU_TABLE[0, :], ITU_TABLE[1, :], kind='cubic')
        f_b = interp1d(ITU_TABLE[0, :], ITU_TABLE[3, :], kind='cubic')
    else:
        raise ValueError('Polarization must be 0 (horizontal) or 1 (vertical).')
    a = f_a(frequncey)
    b = f_b(frequncey)
    return a, b
