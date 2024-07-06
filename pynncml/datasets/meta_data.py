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

    def has_location(self):
        return self.lon_lat_site_one is not None and self.lon_lat_site_zero is not None

    def xy(self):
        if self.has_location():
            return np.stack([self.xy_zero, self.xy_one]).flatten()
        else:
            raise Exception("")

    def as_tensor(self) -> torch.Tensor:
        return torch.Tensor(
            [self.height_far, self.height_near, self.frequency, self.polarization, self.length]).reshape(1, -1).float()

    def xy_center(self):
        return (self.xy_zero[0] + self.xy_one[0]) / 2, (self.xy_zero[1] + self.xy_one[1]) / 2
