import numpy as np
import math

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

    def has_tsl(self):
        raise NotImplemented

    def plot(self):
        raise NotImplemented

    def plot_gauge(self):
        raise NotImplemented

    def step(self):
        return np.diff(self.time_array).min() / HOUR_IN_SECONDS

    def cumulative_rain(self):
        return np.cumsum(self.rain_gauge) * self.step()


class Link(LinkBase):
    def __init__(self, link_rsl, rain_gauge, time_array, link_tsl=None):
        """
        Link object is a data structure that contains the link dynamic information:
        recvied signal level (RSL) and trasminted signal level (TSL).
        Addainly this object contatins a refernce sequnce of a near rain gauge.
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

    def create_min_max_link(self, step_size):
        n_steps = math.ceil(self.step() / step_size)



class LinkMinMax(LinkBase):
    def __init__(self, min_rsl, max_rsl, rain_gauge, time_array, min_tsl=None, max_tsl=None):
        super().__init__(time_array, rain_gauge)


def read_cmlh5():
    pass


def read_pickle():
    pass
