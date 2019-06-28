import numpy as np
import pickle
import os
import torch
import torchrain as tr
from abc import abstractstaticmethod
from matplotlib import pyplot as plt

HOUR_IN_SECONDS = 3600


class MetaData(object):
    def __init__(self, frequency, polarization, length, height_far, height_near):
        self.frequency = frequency
        self.polarization = polarization
        self.length = length
        self.height_far = height_far
        self.height_near = height_near


class LinkBase(object):
    def __init__(self, time_array: np.ndarray, rain_gauge: np.ndarray, meta_data: MetaData):
        self._check_input(time_array)
        self._check_input(rain_gauge)
        assert time_array.shape[0] == rain_gauge.shape[0]
        self.rain_gauge = rain_gauge
        self.time = time_array.astype('datetime64[s]')
        self.meta_data = meta_data

    @staticmethod
    def _check_input(input_array):
        assert isinstance(input_array, np.ndarray)
        assert len(input_array.shape) == 1

    def __len__(self):
        return len(self.time)

    @abstractstaticmethod
    def has_tsl(self):
        pass

    @abstractstaticmethod
    def plot_link(self):
        pass

    def plot_gauge(self):
        raise NotImplemented

    def step(self):
        return np.diff(self.time).min() / HOUR_IN_SECONDS

    def cumulative_rain(self):
        return np.cumsum(self.rain_gauge) * self.step()

    def start_time(self):
        return self.time[0]

    def stop_time(self):
        return self.time[-1]

    def delta_time(self):
        return self.stop_time() - self.start_time()


class Link(LinkBase):
    def __init__(self, link_rsl: np.ndarray, rain_gauge: np.ndarray, time_array: np.ndarray, meta_data: MetaData,
                 link_tsl=None):
        """
        Link object is a data structure that contains the link dynamic information:
        received signal level (RSL) and transmitted signal level (TSL).
        Addainly this object contains a refernce sequnce of a near rain gauge.

        :param link_rsl:
        :param rain_gauge:
        :param time_array:
        :param meta_data:
        :param link_tsl:
        """
        super().__init__(time_array, rain_gauge, meta_data)
        self._check_input(link_rsl)
        assert len(link_rsl) == len(self)
        if link_tsl is not None:  # if link tsl is not none check that is valid
            self._check_input(link_tsl)
            assert len(link_tsl) == len(self)
        self.link_rsl = link_rsl
        self.link_tsl = link_tsl

    def plot(self):
        plt.subplot(1, 2, 1)
        plt.plot(self.time, self.attenuation().numpy().flatten())
        plt.ylabel(r'$A_n$')
        plt.title('Attenuation')
        tr.change_x_axis_time_format('%H')
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(self.time, self.rain_gauge)
        plt.ylabel(r'$R_n$')
        tr.change_x_axis_time_format('%H')
        plt.title('Rain')
        plt.grid()

        plt.show()

    def attenuation(self):
        if self.has_tsl():
            return torch.tensor(-(self.tsl - self.rsl)).reshape(1, -1).float()
        else:
            return torch.tensor(-self.link_rsl).reshape(1, -1).float()

    def has_tsl(self):
        return self.link_tsl is not None

    def create_min_max_link(self, step_size):
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
            rsl = self.link_rsl[(self.time >= lt) * (self.time < ht)]
            if self.link_tsl is not None:
                tsl = self.link_tsl[(self.time >= lt) * (self.time < ht)]
            time_vector.append(lt)
            min_rsl_vector.append(rsl.min())
            max_rsl_vector.append(rsl.max())
            rain_vector.append(self.rain_gauge[(self.time >= lt) * (self.time < ht)].mean())
        min_rsl_vector = np.asarray(min_rsl_vector)
        max_rsl_vector = np.asarray(max_rsl_vector)
        min_tsl_vector = np.asarray(min_tsl_vector)
        max_tsl_vector = np.asarray(max_tsl_vector)
        rain_vector = np.asarray(rain_vector)
        time_vector = np.asarray(time_vector)
        if self.link_tsl is not None:
            return LinkMinMax(min_rsl_vector, max_rsl_vector, rain_vector, time_vector, self.meta_data)
        else:
            return LinkMinMax(min_rsl_vector, max_rsl_vector, rain_vector, time_vector, self.meta_data,
                              min_tsl=min_tsl_vector, max_tsl=max_tsl_vector)


class LinkMinMax(LinkBase):
    def __init__(self, min_rsl, max_rsl, rain_gauge, time_array, meta_data: MetaData, min_tsl=None, max_tsl=None):
        super().__init__(time_array, rain_gauge, meta_data)
        self.min_rsl = min_rsl
        self.max_rsl = max_rsl
        self.min_tsl = min_tsl
        self.max_tsl = max_tsl


def read_open_cml_dataset(pickle_path: str):
    if not os.path.isfile(pickle_path):
        raise Exception('The input path: ' + pickle_path + ' is not a file')
    open_cml_ds = pickle.load(open(pickle_path, "rb"))
    return [Link(oc[0], oc[1], oc[2], oc[3]) for oc in open_cml_ds if len(oc) == 4]
