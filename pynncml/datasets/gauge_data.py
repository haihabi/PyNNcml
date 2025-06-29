import utm
import numpy as np

from pynncml.datasets.base_rain_sensors import BaseRainSensor


class PointSensor(BaseRainSensor):
    def __init__(self, data_array: np.ndarray, time_array: np.ndarray, lon: float, lat: float, force_zone_number=32,
                 force_zone_letter="V"):
        """
        Point sensor data class for the gauge data.
        :param data_array: np.ndarray
        :param time_array: np.ndarray
        :param lon: float
        :param lat: float

        """
        super().__init__(data_array, time_array)
        self.lon = lon
        self.lat = lat
        self.force_zone_number = force_zone_number
        self.force_zone_letter = force_zone_letter
        self.x, self.y = utm.from_latlon(self.lat, self.lon, force_zone_number=force_zone_number,
                                         force_zone_letter=force_zone_letter)[:2]

    def change_time_base(self, new_time_base: int) -> 'PointSensor':
        """
        Change the time base of the data array
        :param new_time_base: int
        :return: PointSensor
        """
        data, time = self._resample_array(new_time_base)
        return PointSensor(data, time, self.lon, self.lat,self.force_zone_number,self.force_zone_letter)
