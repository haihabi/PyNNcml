import numpy as np
import pickle
import os
import torch
from dataclasses import dataclass

import pynncml as pnc
from matplotlib import pyplot as plt
from pynncml.datasets.meta_data import MetaData
from enum import Enum

HOUR_IN_SECONDS = 3600


class LinkBase(object):
    def __init__(self,
                 time_array: np.ndarray,
                 rain_gauge: np.ndarray,
                 meta_data: MetaData,
                 gauge_ref=None):
        """
        LinkBase object is a data structure that contains the link dynamic information:
        :param time_array: Time array
        :param rain_gauge: Rain gauge data
        :param meta_data: MetaData object
        :param gauge_ref: Gauge reference
        """
        self._check_input(time_array)
        self.gauge_ref = gauge_ref
        if rain_gauge is not None:
            self._check_input(rain_gauge)
            assert time_array.shape[0] == rain_gauge.shape[0]
        self.rain_gauge = rain_gauge
        self.time_array = time_array
        self.meta_data: MetaData = meta_data

    def plot_link_position(self):
        """
        Plot the link position
        """
        if self.meta_data.has_location():
            xy_array = self.meta_data.xy()
            return xy_array

    def time(self) -> np.ndarray:
        """
        Return the time array as datetime64
        """
        return self.time_array.astype('datetime64[s]')

    @staticmethod
    def _check_input(input_array):
        """
        Check the input array is a numpy array and has one dimension
        """
        assert isinstance(input_array, np.ndarray)
        assert len(input_array.shape) == 1

    def __len__(self) -> int:
        """
        Return the length of the time array
        """
        return len(self.time_array)

    def step(self):
        """
        Return the time step of the time array in hours
        """
        return np.diff(self.time_array).min() / HOUR_IN_SECONDS

    def cumulative_rain(self):
        """
        Return the cumulative rain gauge
        """
        return np.cumsum(self.rain_gauge) * self.step()

    def rain(self):
        """
        Return the rain gauge
        """
        return self.rain_gauge.copy()

    def start_time(self):
        """
        Return the start time of the time array
        """
        return self.time_array[0]

    def stop_time(self):
        return self.time_array[-1]

    def delta_time(self):
        """
        Return the delta time of the time array
        """
        return self.stop_time() - self.start_time()


class LinkMinMax(LinkBase):
    def __init__(self,
                 min_rsl,
                 max_rsl,
                 rain_gauge,
                 time_array,
                 meta_data,
                 min_tsl=None,
                 max_tsl=None,
                 gauge_ref=None):
        super().__init__(time_array, rain_gauge, meta_data, gauge_ref=gauge_ref)
        """
        LinkMinMax object is a data structure that contains the link dynamic information in min max format.
        :param min_rsl: Minimum received signal level
        :param max_rsl: Maximum received signal level
        :param rain_gauge: Rain gauge data
        :param time_array: Time array
        :param meta_data: MetaData object
        :param min_tsl: Minimum transmitted signal level
        :param max_tsl: Maximum transmitted signal level
        :param gauge_ref: Gauge reference
        
        """
        self.min_rsl = min_rsl
        self.max_rsl = max_rsl
        self.min_tsl = min_tsl
        self.max_tsl = max_tsl

    def has_tsl(self) -> bool:
        """
        Check if the link has transmitted signal level
        """
        return self.min_tsl is not None and self.max_tsl is not None

    def attenuation(self) -> torch.Tensor:
        """
        Calculate the attenuation from the link data
        :return attenuation: torch.Tensor
        """
        if self.has_tsl():
            att_min = torch.tensor(self.min_tsl - self.max_rsl).reshape(1, -1, 1).float()
            att_max = torch.tensor((self.max_tsl - self.min_rsl)).reshape(1, -1, 1).float()
        else:
            att_min = torch.tensor(- self.max_rsl).reshape(1, -1, 1).float()
            att_max = torch.tensor(- self.min_rsl).reshape(1, -1, 1).float()
        return torch.cat([att_max, att_min], dim=-1)  # [B, T, 2]

    def plot(self):
        """
        Plot the attenuation and rain gauge data.

        """
        att = self.attenuation()
        att_max = att[0, :, 1]
        att_min = att[0, :, 0]
        if self.rain_gauge is not None: plt.subplot(1, 2, 1)
        plt.plot(self.time(), att_max.numpy().flatten(), label=r'$A_n^{max}$')
        plt.plot(self.time(), att_min.numpy().flatten(), label=r'$A_n^{min}$')
        plt.legend()
        plt.ylabel(r'$A[dB]$')
        plt.title('Attenuation')
        pnc.change_x_axis_time_format('%H')
        plt.grid()
        if self.rain_gauge is not None:
            plt.subplot(1, 2, 2)
            plt.plot(self.time(), self.rain_gauge)
            plt.ylabel(r'$R_n[mm/hr]$')
            pnc.change_x_axis_time_format('%H')
            plt.title('Rain')
            plt.grid()

    def as_tensor(self, constant_tsl=None):
        """
        Return the link data as tensor format.
        :param constant_tsl: Constant transmitted signal level
        :return: torch.Tensor
        """
        if self.has_tsl():
            return torch.stack([torch.Tensor(self.max_rsl).float(), torch.Tensor(self.min_rsl).float(),
                                torch.Tensor(self.max_tsl).float(), torch.Tensor(self.min_tsl).float()])
        else:
            if constant_tsl is None:
                return torch.stack([torch.Tensor(self.max_rsl).float(), torch.Tensor(self.min_rsl).float()])
            else:
                tsl = torch.Tensor(constant_tsl * np.ones(len(self))).float()
                return torch.stack(
                    [tsl, torch.Tensor(self.min_rsl).float(), tsl, torch.Tensor(self.max_rsl).float()],
                    dim=1)


class Link(LinkBase):
    def __init__(self, link_rsl: np.ndarray, time_array: np.ndarray, meta_data,
                 rain_gauge: np.ndarray = None,
                 link_tsl=None,
                 gauge_ref=None):
        """
        Link object is a data structure that contains the link dynamic information:
        received signal level (RSL) and transmitted signal level (TSL).

        :param link_rsl: Received signal level
        :param time_array: Time array
        :param meta_data: MetaData object
        :param link_tsl: Transmitted signal level
        :param rain_gauge: Rain gauge data
        :param gauge_ref: Gauge reference
        """
        super().__init__(time_array, rain_gauge, meta_data, gauge_ref=gauge_ref)
        self._check_input(link_rsl)
        assert len(link_rsl) == len(self)
        if link_tsl is not None:  # if link tsl is not none check that is valid
            self._check_input(link_tsl)
            assert len(link_tsl) == len(self)
        self.link_rsl = link_rsl
        self.link_tsl = link_tsl

    def data_alignment(self):
        """
        Align the link data with the gauge data
        :return: gauge_data, rsl, tsl, meta_data
        """
        delta_gauge = np.min(np.diff(self.gauge_ref.time_array))
        delta_link = np.min(np.diff(self.time_array))

        ratio = int(delta_gauge / delta_link)
        gauge_end_cut = (self.time_array[-1] - self.time_array[-1] % delta_gauge) in self.gauge_ref.time_array
        gauge_start_cut = (self.time_array[0] - self.time_array[0] % delta_gauge) in self.gauge_ref.time_array

        link_end_cut = (self.gauge_ref.time_array[-1] - self.gauge_ref.time_array[-1] % delta_link) in self.time_array
        link_start_cut = (self.gauge_ref.time_array[0] - self.gauge_ref.time_array[0] % delta_link) in self.time_array

        rsl = self.link_rsl
        tsl = self.link_tsl
        time_link = self.time_array
        gauge_data = self.gauge_ref.data_array
        if gauge_start_cut:
            raise NotImplemented

        if gauge_end_cut:
            link_end_point = self.time_array[-1] - self.time_array[-1] % delta_gauge
            i = np.where(self.gauge_ref.time_array == link_end_point)[0][0]
            gauge_data = gauge_data[:(i + 1)]

        if link_start_cut:
            gauge_start_point = self.gauge_ref.time_array[0] - self.gauge_ref.time_array[0] % delta_link
            i = np.where(time_link == gauge_start_point)[0][0]
            rsl = rsl[i:]
            tsl = tsl[i:]
            time_link = time_link[i:]

        if link_end_cut:
            gauge_end_point = self.gauge_ref.time_array[-1] - self.gauge_ref.time_array[-1] % delta_link
            i = np.where(time_link == gauge_end_point)[0][0]
            rsl = rsl[:(i + ratio)]
            tsl = tsl[:(i + ratio)]

        rsl = np.lib.stride_tricks.as_strided(rsl, shape=(int(rsl.shape[0] / ratio), ratio), strides=(4 * ratio, 4))
        tsl = np.lib.stride_tricks.as_strided(tsl, shape=(int(tsl.shape[0] / ratio), ratio), strides=(4 * ratio, 4))

        return gauge_data, rsl, tsl, np.asarray([self.meta_data.frequency, self.meta_data.length]).astype("float32")

    def plot(self):
        """
        Plot the attenuation and rain gauge data.

        """
        if self.rain_gauge is not None: plt.subplot(1, 2, 1)
        plt.plot(self.time(), self.attenuation().numpy().flatten())
        plt.ylabel(r'$A_n$')
        plt.title('Attenuation')
        pnc.change_x_axis_time_format('%H')
        plt.grid()
        if self.rain_gauge is not None:
            plt.subplot(1, 2, 2)
            plt.plot(self.time(), self.rain_gauge)
            plt.ylabel(r'$R_n$')
            pnc.change_x_axis_time_format('%H')
            plt.title('Rain')
            plt.grid()

    def attenuation(self) -> torch.Tensor:
        """
        Calculate the attenuation from the link data
        :return attenuation: torch.Tensor
        """
        if self.has_tsl():
            return torch.tensor(-(self.link_tsl - self.link_rsl)).reshape(1, -1).float()
        else:
            return torch.tensor(-self.link_rsl).reshape(1, -1).float()

    def has_tsl(self) -> bool:
        """
        Check if the link has transmitted signal level

        """
        return self.link_tsl is not None

    def create_min_max_link(self, step_size) -> LinkMinMax:
        """
        Create a min max link from the link data
        :param step_size: Step size
        """
        low_time = np.linspace(self.start_time(), self.stop_time() - step_size,
                               np.ceil(self.delta_time() / step_size).astype('int'))
        high_time = np.linspace(self.start_time() + step_size, self.stop_time(),
                                np.ceil(self.delta_time() / step_size).astype('int'))
        time_vector = []
        min_rsl_vector = []
        min_tsl_vector = []
        max_tsl_vector = []
        max_rsl_vector = []
        rain_vector = []
        for lt, ht in zip(low_time, high_time):  # loop over high and low time step
            rsl = self.link_rsl[(self.time_array >= lt) * (self.time_array < ht)]
            min_rsl_vector.append(rsl.min())
            max_rsl_vector.append(rsl.max())
            if self.link_tsl is not None:
                tsl = self.link_tsl[(self.time_array >= lt) * (self.time_array < ht)]
                min_tsl_vector.append(tsl.min())
                max_tsl_vector.append(tsl.max())
            time_vector.append(lt)

            if self.rain_gauge is not None:
                rain_vector.append(self.rain_gauge[(self.time_array >= lt) * (self.time_array < ht)].mean())
        min_rsl_vector = np.asarray(min_rsl_vector)
        max_rsl_vector = np.asarray(max_rsl_vector)
        min_tsl_vector = np.asarray(min_tsl_vector)
        max_tsl_vector = np.asarray(max_tsl_vector)
        if self.rain_gauge is not None:
            rain_vector = np.asarray(rain_vector)
        else:
            rain_vector = None
        time_vector = np.asarray(time_vector)
        if self.has_tsl():
            return LinkMinMax(min_rsl_vector, max_rsl_vector, rain_vector, time_vector, self.meta_data,
                              min_tsl=min_tsl_vector, max_tsl=max_tsl_vector, gauge_ref=self.gauge_ref)
        else:
            return LinkMinMax(min_rsl_vector, max_rsl_vector, rain_vector, time_vector, self.meta_data,
                              gauge_ref=self.gauge_ref)


# TODO:Remove this function and replace with OpenMRG dataset
def read_open_cml_dataset(pickle_path: str) -> list:
    if not os.path.isfile(pickle_path):
        raise Exception('The input path: ' + pickle_path + ' is not a file')
    with open(pickle_path, "rb") as f:
        open_cml_ds = pickle.load(f)
    return [Link(oc[0], oc[1], oc[2], oc[3]) for oc in open_cml_ds if len(oc) == 4]


class AttenuationType(Enum):
    """
    Attenuation type enumeration

    """
    MIN_MAX = 'min_max'
    REGULAR = 'regular'


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
        attenuation_type = AttenuationType.REGULAR
    elif len(attenuation.shape) == 3 and attenuation.shape[2] == 2:
        att_max, att_min = attenuation[:, :, 0], attenuation[:, :, 1]  # split the attenuation to max and min
        attenuation_type = AttenuationType.MIN_MAX
    else:
        raise Exception('The input attenuation vector dont match min max format or regular format')
    return AttenuationData(attenuation_min=att_min,
                           attenuation_max=att_max,
                           attenuation=attenuation_avg,
                           attenuation_type=attenuation_type)
