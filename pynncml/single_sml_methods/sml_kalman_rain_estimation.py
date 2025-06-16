import numpy as np

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
dh = 0.400
h0 = 2

def sml_rain_estimator(sat_data, preprocess=False):
    din_r = sat_data.esno.to_numpy()
    if preprocess:
        din_r = sat_data_preprocessing(din_r)

    ST = apply_kalman_filter_st(din_r)
    FT = apply_kalman_filter_ft(din_r)

    [FT, ST, Diff_Rain_Flag] = apply_st_mask(FT, ST)
    L = Calc_L(FT, ST)
    ####################  Apply power Law  ##############################################
    RainRate_org = ApplyPowerLaw(L)
    RainEstimated= ApplyDiffMask(RainRate_org, Diff_Rain_Flag)

    return RainEstimated, ST, FT


def apply_st_mask(FT, ST):
    Diff = ST - FT

    Diff_Rain_Flag = np.zeros((len(Diff)))
    Diff_Rain_Flag[Diff >= DiffGapValue] = 1
    z2one_ind = np.where(np.roll(Diff_Rain_Flag, 1) != Diff_Rain_Flag)[0]

    z2one_ind = z2one_ind[1:len(z2one_ind) - 1]
    z2onemat = np.reshape(z2one_ind, (-1, 2))

    for ind in z2onemat:
        ST[ind[0]:ind[1]] = ST[ind[0] + 1]

    return FT, ST, Diff_Rain_Flag


def Calc_L(FT, ST):
    L = (np.multiply(np.divide(FT, ST), (Tc / La + Tm * (1 - 1 / La) + Tg + Tr)) + (Tm - Tc) / La) / (Tm + Tg + Tr)
    return L


def sat_data_preprocessing(in_data, timedelta=0.5, interval=15):
    indexes = np.ceil(interval / timedelta)
    Din_r = np.zeros(len(in_data))
    for x in range(in_data.shape[0]):
        if x > 0:
            if x <= indexes:
                Din_r[x] = -1 * (np.max(in_data[0:x]) - in_data[x])
            else:
                Din_r[x] = -1 * (np.max(in_data[x - int(indexes):x]) - in_data[x])
        else:
            Din_r[x] = in_data[x]
    Din_r = Din_r + np.median(in_data)
    return Din_r


def ApplyDiffMask(RainRate_org, Diff_Rain_Flag):
    RainRate = np.multiply(RainRate_org, Diff_Rain_Flag)
    RainEstimated = RainRate[~np.isnan(np.nan_to_num(RainRate,
                                                     0))]  # 120 - normelize rain rate to mm/h ( we have 120 samples of 30 second in one hour  )

    return RainEstimated


def ApplyPowerLaw(L):
    L_dB = 10 * np.log10(L)
    hr = h0 - dh
    L1_dB = L_dB * np.sin(teta) / hr
    RainRate_org = alpha * (np.power(L1_dB, beta, dtype=complex))
    RainRate_org = abs(RainRate_org)
    return (RainRate_org)
