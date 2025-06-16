import numpy as np
import torch
import math
from pynncml.single_sml_methods.kalman_filter import apply_kalman_filter_ft, apply_kalman_filter_st

DiffGapValue = 0.4
Tm = 275
Tg = 50
Tc = 3
La = 0.3
Tr = 14
alpha = 28.14  # 17.910
beta = 0.798
teta = 31.58 * np.pi / 180
teta_sin = float(np.sin(teta))
dh = 0.400
h0 = 2


def sml_rain_estimator(sat_data: torch.Tensor, preprocess=False):
    """
    Estimates rain rate from satellite data using Kalman filtering and power law.
    :param sat_data: A 2D tensor of shape (n_samples, timesteps) containing satellite data.
    :param preprocess: If True, applies preprocessing to the satellite data.
    :return: A tuple containing the estimated rain rate, ST, and FT.
    """
    if not isinstance(sat_data, torch.Tensor):
        raise TypeError("sat_data must be a torch.Tensor")
    if len(sat_data.shape) != 2:
        raise ValueError("sat_data must be a 2D tensor with shape (n_samples, timesteps)")
    if sat_data.shape[1] < 2:
        raise ValueError("sat_data must have at least 2 timesteps")
    if sat_data.shape[0] < 1:
        raise ValueError("sat_data must have at least 1 sample")
    if sat_data.dtype != torch.float32 and sat_data.dtype != torch.float64:
        raise TypeError("sat_data must be of type torch.float32 or torch.float64")

    din_r = sat_data  # sat_data.esno.to_numpy()
    if preprocess:
        din_r = sat_data_preprocessing(din_r)
    ST = apply_kalman_filter_st(din_r)
    FT = apply_kalman_filter_ft(din_r)

    [FT, ST, Diff_Rain_Flag] = apply_st_mask(FT, ST)
    L = calc_l(FT, ST)

    ####################  Apply power Law  ##############################################
    RainRate_org = apply_power_law(L)
    RainEstimated = ApplyDiffMask(RainRate_org, Diff_Rain_Flag)
    return RainEstimated, ST, FT


def apply_st_mask(FT: torch.Tensor, ST: torch.Tensor):
    delta = ST - FT

    diff_rain_ind = torch.zeros(delta.shape)
    diff_rain_ind[delta >= DiffGapValue] = 1
    sample, z2one_ind = torch.where(torch.roll(diff_rain_ind, 1, dims=1) != diff_rain_ind)
    for s in torch.unique(sample):
        _z2one_ind= z2one_ind[sample == s]
        _z2one_ind = _z2one_ind[1:len(_z2one_ind) - 1]
        z2onemat = torch.reshape(_z2one_ind, (-1, 2))
        for ind in z2onemat:
            ST[s, ind[0]:ind[1]] = ST[s, ind[0] + 1]

    return FT, ST, diff_rain_ind


def calc_l(FT: torch.Tensor, ST: torch.Tensor):
    L = (torch.multiply(torch.divide(FT, ST), (Tc / La + Tm * (1 - 1 / La) + Tg + Tr)) + (Tm - Tc) / La) / (
            Tm + Tg + Tr)
    return L


def sat_data_preprocessing(in_data: torch.Tensor, timedelta=0.5, interval=15):
    indexes = math.ceil(interval / timedelta)
    din_r = torch.zeros(in_data.shape)
    for ti in range(in_data.shape[1]):
        if ti > 0:
            if ti <= indexes:
                din_r[:, ti] = -1 * (torch.max(in_data[:, 0:ti], dim=1)[0] - in_data[:, ti])
            else:
                din_r[:, ti] = -1 * (torch.max(in_data[:, ti - int(indexes):ti], dim=1)[0] - in_data[:, ti])
        else:
            din_r[:, ti] = in_data[:, ti]
    din_r = din_r + torch.median(in_data, dim=1).values
    return din_r


def ApplyDiffMask(RainRate_org, Diff_Rain_Flag):
    rain_rate = torch.multiply(RainRate_org, Diff_Rain_Flag)

    rain_estimated = torch.stack(
        [rain_rate[i, ~torch.isnan(torch.nan_to_num(rain_rate[i,:], 0))] for i in range(rain_rate.shape[0])],
        dim=0)  # 120 - normelize rain rate to mm/h ( we have 120 samples of 30 second in one hour  )
    return rain_estimated


def apply_power_law(L: torch.Tensor):
    L_dB = 10 * torch.log10(L)
    hr = h0 - dh
    L1_dB = L_dB * teta_sin / hr
    L1_dB_complex=torch.complex(L1_dB,torch.zeros_like(L1_dB))
    RainRate_org = alpha * (torch.pow(L1_dB_complex, beta))
    RainRate_org = torch.abs(RainRate_org)
    return RainRate_org
