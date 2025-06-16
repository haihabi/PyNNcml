import numpy as np


def apply_kalman_filter(observations, r_scale=1 / 1000, q=None):
    # intial parameters

    z = observations
    n_iter = len(z)
    sz = (n_iter,)  # size of array

    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor
    R = np.var(z) * r_scale
    Q = np.var(z) / 1000 if q is None else q  # 1e-5 process variance
    # intial guesses
    xhat[0] = np.median(z)
    P[0] = 1.0

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat


def apply_kalman_filter_ft(observations):
    """
    Apply a Kalman filter to the observations.
    :param observations: The input observations.
    :param r_scale: Scaling factor for the measurement noise variance.
    :param q: Process noise variance. If None, it will be set to a default value.
    :return: Filtered observations.
    """
    return apply_kalman_filter(observations, r_scale=1/1000, q=1e-5)


def apply_kalman_filter_st(observations):
    """
    Apply a Kalman filter to the observations.
    :param observations: The input observations.
    :param r_scale: Scaling factor for the measurement noise variance.
    :param q: Process noise variance. If None, it will be set to a default value.
    :return: Filtered observations.
    """
    return apply_kalman_filter(observations, r_scale=100, q=None)
