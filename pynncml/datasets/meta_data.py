import torch
import utm
import numpy as np
from typing import List


class MetaData(object):
    def __init__(self,
                 frequency: float,
                 polarization: bool,
                 length: float,
                 height_far: float,
                 height_near: float,
                 lon_lat_site_zero: List[float] = None,
                 lon_lat_site_one: List[float] = None):
        self.frequency = frequency
        self.polarization = polarization
        self.length = length
        self.height_far = height_far
        self.height_near = height_near
        self.lon_lat_site_zero = lon_lat_site_zero
        self.lon_lat_site_one = lon_lat_site_one
        if self.has_location():
            self.xy_zero = np.flip(utm.from_latlon(self.lon_lat_site_zero[0], self.lon_lat_site_zero[1])[:2])
            self.xy_one = np.flip(utm.from_latlon(self.lon_lat_site_one[0], self.lon_lat_site_one[1])[:2])
            self.xy_scale_zero = None
            self.xy_scale_one = None

    def has_location(self):
        return self.lon_lat_site_one is not None and self.lon_lat_site_zero is not None

    def has_scale(self):
        return self.xy_scale_zero is not None and self.xy_scale_one is not None

    def update_scale(self, x_min, x_delta, y_min, y_delta):
        self.xy_scale_zero = [(self.xy_zero[0] - x_min) / x_delta, (self.xy_zero[1] - y_min) / y_delta]
        self.xy_scale_one = [(self.xy_one[0] - x_min) / x_delta, (self.xy_one[1] - y_min) / y_delta]

    def xy(self):
        if self.has_location():
            return np.stack([self.xy_zero, self.xy_one]).flatten()
        else:
            raise Exception("")

    def xy_scale(self):
        if self.has_scale():
            return np.stack([self.xy_scale_zero, self.xy_scale_one]).flatten()
        else:
            raise Exception("")

    def as_tensor(self) -> torch.Tensor:
        return torch.Tensor(
            [self.height_far, self.height_near, self.frequency, self.polarization, self.length]).reshape(1, -1).float()

    def xy_center(self):
        if self.has_scale():
            return (self.xy_scale_zero[0] + self.xy_scale_one[0]) / 2, (
                    self.xy_scale_zero[1] + self.xy_scale_one[1]) / 2
        else:
            return (self.xy_zero[0] + self.xy_one[0]) / 2, (self.xy_zero[1] + self.xy_one[1]) / 2
