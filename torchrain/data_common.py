import numpy as np
from abc import abstractstaticmethod

HOUR_IN_SECONDS = 3600


class LinkBase(object):
    def __init__(self, time_array, rain_gauge):
        self._check_input(time_array)
        self._check_input(rain_gauge)
        assert time_array.shape[0] == rain_gauge.shape[0]
        self.rain_gauge = rain_gauge
        self.time_array = time_array

    @staticmethod
    def _check_input(input_array):
        assert isinstance(input_array, np.ndarray)
        assert len(input_array.shape) == 1

    def __len__(self):
        return len(self.time_array)

    @abstractstaticmethod
    def has_tsl(self):
        pass

    @abstractstaticmethod
    def plot_link(self):
        pass

    def plot_gauge(self):
        raise NotImplemented

    def step(self):
        return np.diff(self.time_array).min() / HOUR_IN_SECONDS

    def cumulative_rain(self):
        return np.cumsum(self.rain_gauge) * self.step()

    def start_time(self):
        return self.time_array[0]

    def stop_time(self):
        return self.time_array[-1]

    def delta_time(self):
        return self.stop_time() - self.start_time()


class Link(LinkBase):
    def __init__(self, link_rsl: np.ndarray, rain_gauge: np.ndarray, time_array: np.ndarray, link_tsl=None):
        """
        Link object is a data structure that contains the link dynamic information:
        received signal level (RSL) and transmitted signal level (TSL).
        Addainly this object contains a refernce sequnce of a near rain gauge.

        :param link_rsl:
        :param rain_gauge:
        :param time_array:
        :param link_tsl:
        """
        super().__init__(time_array, rain_gauge)
        self._check_input(link_rsl)
        assert len(link_rsl) == len(self)
        if link_tsl is not None:  # if link tsl is not none check that is valid
            self._check_input(link_tsl)
            assert len(link_tsl) == len(self)
        self.link_rsl = link_rsl
        self.link_tsl = link_tsl

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
            rsl = self.link_rsl[(self.time_array >= lt) * (self.time_array < ht)]
            if self.link_tsl is not None:
                tsl = self.link_tsl[(self.time_array >= lt) * (self.time_array < ht)]
            time_vector.append(lt)
            min_rsl_vector.append(rsl.min())
            max_rsl_vector.append(rsl.max())
            rain_vector.append(self.rain_gauge[(self.time_array >= lt) * (self.time_array < ht)].mean())
        min_rsl_vector = np.asarray(min_rsl_vector)
        max_rsl_vector = np.asarray(max_rsl_vector)
        min_tsl_vector = np.asarray(min_tsl_vector)
        max_tsl_vector = np.asarray(max_tsl_vector)
        rain_vector = np.asarray(rain_vector)
        time_vector = np.asarray(time_vector)
        return LinkMinMax(min_rsl_vector, max_rsl_vector, rain_vector, time_vector)


class LinkMinMax(LinkBase):
    def __init__(self, min_rsl, max_rsl, rain_gauge, time_array, min_tsl=None, max_tsl=None):
        super().__init__(time_array, rain_gauge)
        self.min_rsl = min_rsl
        self.max_rsl = max_rsl
        self.min_tsl = min_tsl
        self.max_tsl = max_tsl


def read_cmlh5():
    pass


def read_pickle():
    pass
