import numpy as np
import torch
from torch import nn


class KalmanFilter(nn.Module):
    def __init__(self, r_scale, q=None):
        """
        Initialize the Kalman filter.
        :param r_scale: Scaling factor for the measurement noise variance.
        :param q: Process noise variance. If None, it will be set to a default value.
        """
        super(KalmanFilter, self).__init__()
        self.r_scale = r_scale
        self.q = q if q is not None else 1e-5

    def forward(self, in_data):
        """
        Apply the Kalman filter to the input data.
        :param in_data: The input data to be filtered [B,T].
        :return: Filtered data.
        """
        xhat = torch.zeros(in_data.shape[0], in_data.shape[1], device=in_data.device)
        P = torch.zeros(in_data.shape[0], in_data.shape[1], device=in_data.device)
        xhatminus = torch.zeros(in_data.shape[0], in_data.shape[1], device=in_data.device)
        Pminus = torch.zeros(in_data.shape[0], in_data.shape[1], device=in_data.device)
        K = torch.zeros(in_data.shape[0], in_data.shape[1], device=in_data.device)
        R = torch.var(in_data, dim=1, keepdim=True) * self.r_scale
        Q = torch.var(in_data, dim=1, keepdim=True) / 1000 if self.q is None else self.q * torch.ones([1],
                                                                                                      device=in_data.device)  # 1e-5 process variance
        # initial guesses
        xhat[:, 0] = torch.median(in_data, dim=1).values
        P[:, 0] = 1.0
        for k in range(1, in_data.shape[1]):
            # time update
            xhatminus[:, k] = xhat[:, k - 1]
            Pminus[:, k] = P[:, k - 1] + Q

            # measurement update
            K[:, k] = Pminus[:, k] / (Pminus[:, k] + R)
            xhat[:, k] = xhatminus[:, k] + K[:, k] * (in_data[:, k] - xhatminus[:, k])
            P[:, k] = (1 - K[:, k]) * Pminus[:, k]
        return xhat


def apply_kalman_filter_ft(observations: torch.Tensor):
    """
    Apply a Kalman filter to the observations.
    :param observations: The input observations.
    :param r_scale: Scaling factor for the measurement noise variance.
    :param q: Process noise variance. If None, it will be set to a default value.
    :return: Filtered observations.
    """
    return KalmanFilter(r_scale=1 / 1000, q=1e-5)(observations)


def apply_kalman_filter_st(observations):
    """
    Apply a Kalman filter to the observations.
    :param observations: The input observations.
    :param r_scale: Scaling factor for the measurement noise variance.
    :param q: Process noise variance. If None, it will be set to a default value.
    :return: Filtered observations.
    """
    return KalmanFilter(r_scale=100, q=None)(observations)
