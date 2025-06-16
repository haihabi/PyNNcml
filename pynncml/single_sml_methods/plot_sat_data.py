import torch
import numpy as np
from matplotlib import pyplot as plt, dates as mdates


def plot_rain_vs_sat_data(rain_array: torch.Tensor, rain_timestamp, sat_array: torch.Tensor, sat_timestamp):
    """
    Plots the rain gauge data against the satellite signal data.
    :param rain_array: 2D tensor of shape [n_samples, timesteps] containing the rain gauge data.
    :param rain_timestamp: Timestamps corresponding to the rain gauge data.
    :param sat_array: 2D tensor of shape [n_samples, timesteps] containing the satellite signal data.
    :param sat_timestamp: Timestamps corresponding to the satellite signal data.
    :return: None
    """
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax1 = ax.twinx()

    ax.plot_date(sat_timestamp, sat_array.cpu().detach().numpy(), '-', color='purple', label='Esno Data ')
    ax1.plot_date(rain_timestamp, rain_array.cpu().detach().numpy(), '-', label="Rain Gauge")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.set_ylabel('Satellite  Signal Es/No [dB]')
    ax1.set_ylabel('Rain[mm/h]')

    ax.legend(loc='upper left')
    ax1.legend(loc='upper right')
    ax.grid()

    fig.suptitle(' Rain Gauge Vs Satellite Signal')
    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()



def plot_rain_estimation_vs_gauge_data(rain_estimation: torch.Tensor,
                                       ST: torch.Tensor,
                                       FT: torch.Tensor,
                                       rain_array: torch.Tensor,
                                       rain_timestamp,
                                       sat_array: torch.Tensor,
                                       sat_timestamp):
    """
    Plots the rain estimation against the rain gauge data and satellite signal data.
    :param rain_estimation: 2D tensor of shape [n_samples, timesteps] containing the rain estimation data.
    :param ST: 2D tensor of shape [n_samples, timesteps] containing the SlowTracker data.
    :param FT: 2D tensor of shape [n_samples, timesteps] containing the FastTracker data.
    :param rain_array: 2D tensor of shape [n_samples, timesteps] containing the rain gauge data.
    :param rain_timestamp: Timestamps corresponding to the rain gauge data.
    :param sat_array: 2D tensor of shape [n_samples, timesteps] containing the satellite signal data.
    :param sat_timestamp: Timestamps corresponding to the satellite signal data.
    :return: None

    """

    if len(rain_array.shape) != 2:
        raise ValueError("rain_array should be a 2D tensor with shape [n_samples, timesteps]")
    if rain_array.shape[1] < 2:
        raise ValueError("rain_array should have at least 2 timesteps")
    if rain_array.shape[0] < 1:
        raise ValueError("rain_array should have at least 1 sample")
    if rain_array.dtype != torch.float32 and rain_array.dtype != torch.float64:
        raise TypeError("rain_array must be of type torch.float32 or torch.float64")

    if len(sat_array.shape) != 2:
        raise ValueError("sat_array should be a 2D tensor with shape [n_samples, timesteps]")
    if sat_array.shape[1] < 2:
        raise ValueError("sat_array should have at least 2 timesteps")
    if sat_array.shape[0] < 1:
        raise ValueError("sat_array should have at least 1 sample")
    if sat_array.dtype != torch.float32 and sat_array.dtype != torch.float64:
        raise TypeError("sat_array must be of type torch.float32 or torch.float64")

    if len(rain_estimation.shape) != 2:
        raise ValueError("rain_estimation should be a 2D tensor with shape [n_samples, timesteps]")
    if rain_estimation.shape[1] < 2:
        raise ValueError("rain_estimation should have at least 2 timesteps")
    if rain_estimation.shape[0] < 1:
        raise ValueError("rain_estimation should have at least 1 sample")
    if rain_estimation.dtype != torch.float32 and rain_estimation.dtype != torch.float64:
        raise TypeError("rain_estimation must be of type torch.float32 or torch.float64")

    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax1 = ax.twinx()

    rain_array_np = rain_array.cpu().detach().numpy()[0, :].flatten()
    sat_array_np = sat_array.cpu().detach().numpy()[0, :].flatten()
    rain_est = rain_estimation.cpu().detach().numpy()[0, :].flatten()
    ST = ST.cpu().detach().numpy()[0, :].flatten()
    FT = FT.cpu().detach().numpy()[0, :].flatten()

    ax.plot_date(sat_timestamp, sat_array_np, '-', color='purple', label='Esno Data ')
    ax.plot_date(sat_timestamp, ST, '-', color='c', label='SlowTracker')
    ax.plot_date(sat_timestamp, FT, '-', color='b', label='FastTracker')
    ax1.plot_date(sat_timestamp, rain_est, '-', color='r', label='Rain Rate')
    ax1.plot_date(sat_timestamp, np.cumsum(rain_est/120), '-', color='k', label='Cumulative Sum Rain Rate')

    ax1.plot_date(rain_timestamp, rain_array_np, '-', color='g', label="Rain Gauge")
    ax1.plot_date(rain_timestamp, np.cumsum(rain_array_np / 12), '-', color='y',
                  label="Cumulative Rain Gauge")
    RainGaugSum = np.sum(rain_array_np / 12)
    EstRainSum = np.sum(rain_est / 120)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # ax1.set_ylim([0, 10])
    # ax.set_ylim([-15, 12])
    ax.set_ylabel('Satellite  Signal Es/No [dB]')

    # ax1.set_ylim([0, 80])
    ax1.set_ylabel('Rain[mm/h]')

    ax.legend(loc='upper left')
    ax1.legend(loc='upper right')
    ax.grid()

    fig.suptitle('satellite link measurements with rain rate of' + str(RainGaugSum) + "   [mm]" +
                 "      |        Sum Algorithm   " + str(EstRainSum) + "   [mm]" + " \n")
    # fig.suptitle(' Rain Gauge Vs Sat Data at Kfar Saba ' + str(RainDF.datetime.iloc[100].date()))
    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()
