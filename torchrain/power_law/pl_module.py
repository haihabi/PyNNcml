from pycomlink.processing.A_R_relation.A_R_relation import ITU_table
from torch import nn
import torch
import numpy as np
from scipy.interpolate import interp1d

ITU_table = np.array([
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


class PowerLaw(nn.Module):
    def __init__(self, r_min):
        super(PowerLaw, self).__init__()
        self.r_min = r_min

    def forward(self, input_attenuation, length, frequncey, polrization):  # model forward pass
        a, b = a_b_parameters(frequncey, polrization)
        beta = 1 / b
        alpha = 1 / (a * length)
        rain_rate = alpha * torch.pow(input_attenuation.float(), beta)
        rain_rate[rain_rate < self.r_min] = 0  # zero rain value below minmal rain
        return rain_rate


class PowerLawMinMax(nn.Module):
    def __init__(self, r_min, k=90):
        super(PowerLawMinMax, self).__init__()
        self.r_min = r_min
        self.k = k

    def forward(self, input_attenuation, length, frequncey, polrization):
        a, b = a_b_parameters(frequncey, polrization)
        a_max = a * (np.log(self.k) + EULER_GAMMA) ** b
        beta = 1 / b
        alpha = np.power(1 / (a_max * length), beta)
        att = input_attenuation * (input_attenuation > 0).float()
        rain_rate = alpha * torch.pow(att, beta)
        rain_rate[rain_rate < self.r_min] = 0  # zero rain value below minmal rain
        return rain_rate


def a_b_parameters(frequncey, polrization):
    """Approximation of parameters for A-R relationship

    Parameters
    ----------
    f_GHz : int, float or np.array of these
            Frequency of the microwave link in GHz
    pol : str
            Polarization of the microwave link

    Returns
    -------
    a,b : float
          Parameters of A-R relationship

    Note
    ----
    The frequency value must be between 1 Ghz and 100 GHz.

    The polarization has to be indicated by 'h' or 'H' for horizontal and
    'v' or 'V' for vertical polarization respectively.

    References
    ----------
    .. [4] ITU, "ITU-R: Specific attenuation model for rain for use in
        prediction methods", International Telecommunication Union, 2013

    """
    frequncey = np.asarray(frequncey)

    if frequncey.min() < 1 or frequncey.max() > 100:
        raise ValueError('Frequency must be between 1 Ghz and 100 GHz.')
    else:
        if polrization == 'V' or polrization == 'v':
            f_a = interp1d(ITU_table[0, :], ITU_table[2, :], kind='cubic')
            f_b = interp1d(ITU_table[0, :], ITU_table[4, :], kind='cubic')
        elif polrization == 'H' or polrization == 'h':
            f_a = interp1d(ITU_table[0, :], ITU_table[1, :], kind='cubic')
            f_b = interp1d(ITU_table[0, :], ITU_table[3, :], kind='cubic')
        else:
            raise ValueError('Polarization must be V, v, H or h.')
        a = f_a(frequncey)
        b = f_b(frequncey)
    return a, b
