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
                 lon_lat_site_one: List[float] = None,
                 force_zone_number=32, force_zone_letter="V"
                 ):
        """
        Metadata class for the links
        :param frequency: Frequency
        :param polarization: Polarization
        :param length: Length
        :param height_far: Height far
        :param height_near: Height near
        :param lon_lat_site_zero: Longitude and latitude of site zero
        :param lon_lat_site_one: Longitude and latitude of site one
        """

        self.frequency = frequency
        self.polarization = polarization
        self.length = length
        self.height_far = height_far
        self.height_near = height_near
        self.lon_lat_site_zero = lon_lat_site_zero
        self.lon_lat_site_one = lon_lat_site_one
        if self.has_location():
            self.xy_zero = utm.from_latlon(self.lon_lat_site_zero[1], self.lon_lat_site_zero[0],force_zone_number=force_zone_number,force_zone_letter=force_zone_letter)[:2]
            self.xy_one = utm.from_latlon(self.lon_lat_site_one[1], self.lon_lat_site_one[0],force_zone_number=force_zone_number,force_zone_letter=force_zone_letter)[:2]

    def has_location(self) -> bool:
        """
        Check if the metadata has location information
        :return: bool
        """
        return self.lon_lat_site_one is not None and self.lon_lat_site_zero is not None

    def xy(self):
        """
        Get the xy coordinates of the metadata
        :return: np.ndarray
        """
        if self.has_location():
            return np.stack([self.xy_zero, self.xy_one]).flatten()
        else:
            raise Exception("No location information")

    def as_tensor(self) -> torch.Tensor:
        """
        Get the metadata as tensor
        :return: torch.Tensor
        """
        return torch.Tensor(
            [self.height_far, self.height_near, self.frequency, self.polarization, self.length]).reshape(1, -1).float()

    def xy_center(self):
        """
        Get the center location of the link
        :return: np.ndarray
        """
        return (self.xy_zero[0] + self.xy_one[0]) / 2, (self.xy_zero[1] + self.xy_one[1]) / 2
