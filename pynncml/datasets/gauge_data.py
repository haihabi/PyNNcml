import utm
import numpy as np


class PointSensor:
    def __init__(self, data_array, time_array, lon, lat):
        self.lon = lon
        self.lat = lat
        self.data_array = data_array
        self.time_array = time_array
        if self.data_array.shape[0] != self.time_array.shape[0]:
            raise Exception("Array shape mismatch")
        self.y, self.x = utm.from_latlon(self.lat, self.lon)[:2]

    def change_time_base(self, new_time_base):
        start_time = self.time_array[0] if self.time_array[0] % new_time_base == 0 else self.time_array[0] + (
                new_time_base - self.time_array[0] % new_time_base)
        end_time = (self.time_array[-1]) if self.time_array[-1] % new_time_base == 0 else self.time_array[-1] - \
                                                                                          self.time_array[
                                                                                              -1] % new_time_base
        end_time -= new_time_base
        ratio = int(new_time_base / np.min(np.diff(self.time_array)))
        time = np.linspace(start_time, end_time, num=int((end_time - start_time) / new_time_base) + 1)
        i_start = np.where(time[0] == self.time_array)[0][0]
        i_end = np.where(time[-1] == self.time_array)[0][0]
        data = self.data_array[i_start:(i_end + ratio)]

        data = np.lib.stride_tricks.as_strided(data, shape=(int(data.shape[0] / ratio), ratio),
                                               strides=(8 * ratio, 8))
        return PointSensor(data.mean(axis=-1), time, self.lon, self.lat)
