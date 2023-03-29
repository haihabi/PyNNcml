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

    def has_location(self):
        return self.lon_lat_site_one is not None and self.lon_lat_site_zero is not None

    def xy(self):
        return np.stack([utm.from_latlon(self.lon_lat_site_zero[1], self.lon_lat_site_zero[0])[:2],
                         utm.from_latlon(self.lon_lat_site_one[1], self.lon_lat_site_one[0])[:2]]).flatten()

    def as_tensor(self) -> torch.Tensor:
        return torch.Tensor(
            [self.height_far, self.height_near, self.frequency, self.polarization, self.length]).reshape(1, -1)


class MetaDataSet(object):
    def __init__(self, meta_data_list: List[MetaData]):
        self.meta_data_list = meta_data_list
