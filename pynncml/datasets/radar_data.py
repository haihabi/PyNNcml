import numpy as np
import utm

from pynncml.datasets.base_rain_sensors import BaseRainSensor



class RadarProjection(BaseRainSensor):
    def __init__(self, data_array: np.ndarray, time_array: np.ndarray, lon_lat_zero: tuple,
                 lon_lat_one: tuple):
        """
        Radar projection data class for the radar data.
        :param data_array: np.ndarray
        :param time_array: np.ndarray
        :param lon_lat_zero: tuple
        :param lon_lat_one: tuple
        """
        super().__init__(data_array, time_array)
        self.lon_lat_zero = lon_lat_zero
        self.lon_lat_one = lon_lat_one

    def change_time_base(self, new_time_base: int) -> 'RadarProjection':
        """
        Change the time base of the data array.
        :param new_time_base: int
        :return: RadarProjection
        """
        data, time = self._resample_array(new_time_base)
        return RadarProjection(data, time, self.lon_lat_zero, self.lon_lat_one)


def is_inside_path(x_center, y_center, patch_size, xy_zero, xy_one):
    x_high = x_center + patch_size / 2.0
    x_low = x_center - patch_size / 2.0
    y_high = y_center + patch_size / 2.0
    y_low = y_center - patch_size / 2.0
    zero_inside = (x_low <= xy_zero[0] <= x_high and y_low <= xy_zero[1] <= y_high)
    one_inside = (x_low <= xy_one[0] <= x_high and y_low <= xy_one[1] <= y_high)
    return zero_inside, one_inside


def find_entry_exit_point(xy_zero, xy_one, x_center, y_center, patch_size):
    x_high = x_center + patch_size / 2.0
    x_low = x_center - patch_size / 2.0
    y_high = y_center + patch_size / 2.0
    y_low = y_center - patch_size / 2.0

    c_1 = xy_zero[1]
    a = (xy_one[1] - xy_zero[1]) / (xy_one[0] - xy_zero[0])
    c_0 = xy_zero[0]

    y_low_exit = c_1 + a * (x_low - c_0)
    y_high_exit = c_1 + a * (x_high - c_0)
    x_low_exit = (y_low - c_1) / a + c_0
    x_high_exit = (y_high - c_1) / a + c_0

    check_y_low = y_low <= y_low_exit <= y_high
    check_y_high = y_low <= y_high_exit <= y_high
    check_x_low = x_low <= x_low_exit <= x_high
    check_x_high = x_low <= x_high_exit <= x_high
    if sum([int(check_y_low), int(check_y_high), int(check_x_low), int(check_x_high)]) != 2:
        raise ValueError("No entry/exit point found within the patch area.")
    points_lis = []
    if check_x_low:
        points_lis.append((x_low_exit, y_low))
    if check_x_high:
        points_lis.append((x_high_exit, y_high))
    if check_y_low:
        points_lis.append((x_low, y_low_exit))
    if check_y_high:
        points_lis.append((x_high, y_high_exit))
    return points_lis


def find_exit_point(xy_zero, xy_one, x_center, y_center, patch_size):
    x_high = x_center + patch_size / 2.0
    x_low = x_center - patch_size / 2.0
    y_high = y_center + patch_size / 2.0
    y_low = y_center - patch_size / 2.0

    c_1 = xy_zero[1]
    a = (xy_one[1] - xy_zero[1]) / (xy_one[0] - xy_zero[0])
    c_0 = xy_zero[0]

    y_low_exit = c_1 + a * (x_low - c_0)
    y_high_exit = c_1 + a * (x_high - c_0)
    x_low_exit = (y_low - c_1) / a + c_0
    x_high_exit = (y_high - c_1) / a + c_0
    # y_high = c_1 + a * (x_high_exist - c_0)

    check_y_low = np.minimum(xy_zero[1], xy_one[1]) <= y_low_exit <= np.maximum(xy_zero[1], xy_one[1])
    check_y_high = np.minimum(xy_zero[1], xy_one[1]) <= y_high_exit <= np.maximum(xy_zero[1], xy_one[1])

    check_x_low = np.minimum(xy_zero[0], xy_one[0]) <= x_low_exit <= np.maximum(xy_zero[0], xy_one[0])
    check_x_high = np.minimum(xy_zero[0], xy_one[0]) <= x_high_exit <= np.maximum(xy_zero[0], xy_one[0])
    if check_x_low:
        x_exit = x_low_exit
        y_exit = y_low
    elif check_x_high:
        x_exit = x_high_exit
        y_exit = y_high
    elif check_y_low:
        x_exit = x_low
        y_exit = y_low_exit
    elif check_y_high:
        x_exit = x_high
        y_exit = y_high_exit
    else:
        raise ValueError("No exit point found within the patch area.")
    return x_exit, y_exit


class RadarData:
    def __init__(self, radar_array, lat_array, lon_array, time_array, force_zone_number=None, force_zone_letter=None):
        """
        Initialize the RadarData class.

        Parameters:
        radar_array (np.ndarray): The radar data array.
        lat_array (np.ndarray): The latitude coordinates.
        lon_array (np.ndarray): The longitude coordinates.
        time_array (np.ndarray): The time coordinates.
        force_zone_number (int, optional): UTM zone number to force.
        force_zone_letter (str, optional): UTM zone letter to force.s
        """
        self.radar_array = radar_array
        self.x, self.y, self.zone_number, self.zone_letter = utm.from_latlon(lat_array, lon_array,
                                                                             force_zone_number=force_zone_number,
                                                                             force_zone_letter=force_zone_letter)
        self.lat_array = lat_array
        self.lon_array = lon_array
        self.time_array = time_array
        self.min_delta = np.min(
            [np.minimum(np.abs(np.diff(self.y, axis=0)).min(), np.abs(np.diff(self.y, axis=1)).min()),
             np.abs(np.diff(self.x, axis=0)).min(), np.abs(np.diff(self.x, axis=1)).min()])

    def _resample_array(self, new_time_base):
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
        data = self.radar_array[i_start:(i_end + ratio), :, :]
        data = np.lib.stride_tricks.as_strided(data,
                                               shape=(int(data.shape[0] / ratio), ratio, data.shape[1], data.shape[2]),
                                               strides=(8 * ratio * data.shape[1] * data.shape[2],
                                                        8 * data.shape[1] * data.shape[2], 8 * data.shape[2], 8))
        data = data.mean(axis=1)
        return data, time

    def change_time_base(self, new_time_base: int) -> 'RadarData':
        """
        Change the time base of the radar data array.

        Parameters:
        new_time_base (int): The new time base in seconds.

        Returns:
        RadarData: A new instance of RadarData with the updated time base.
        """
        data, time = self._resample_array(new_time_base)
        return RadarData(data, self.lat_array, self.lon_array, time)

    def radar_projection2cml(self, lon_lat_zero, lon_lat_one):
        """
        Convert radar data to CML format.

        Parameters:
        lon_lat_zero (tuple): Latitude and longitude of the first site.
        lon_lat_one (tuple): Latitude and longitude of the second site.

        Returns:
        dict: A dictionary containing the radar data in CML format.
        """

        xy_zero = utm.from_latlon(lon_lat_zero[1], lon_lat_zero[0], force_zone_number=self.zone_number,
                                          force_zone_letter=self.zone_letter)[:2]
        xy_one = utm.from_latlon(lon_lat_one[1], lon_lat_one[0], force_zone_number=self.zone_number,
                                         force_zone_letter=self.zone_letter)[:2]
        length_cml = np.sqrt((xy_zero[0] - xy_one[0]) ** 2 + (xy_zero[1] - xy_one[1]) ** 2)
        x_s = np.linspace(xy_zero[0], xy_one[0],
                          np.ceil(np.abs(xy_one[0] - xy_zero[0]) / self.min_delta).astype("int") + 1)
        y_s = xy_zero[1] + (xy_one[1] - xy_zero[1]) * (x_s - xy_zero[0]) / (xy_one[0] - xy_zero[0])
        xy_s = np.array([x_s, y_s]).T
        ind_s = [self.index_location(_xy_s) for _xy_s in xy_s]
        index_list = []
        for i, ind in enumerate(ind_s):
            if ind not in index_list:
                index_list.append(ind)

        patch_size = 2000
        length_size = []
        for ind in index_list:
            # length_size.append(self.min_delta * sum([1 for ind_tag in ind_s if ind == ind_tag]))
            x_center = self.x[ind]
            y_center = self.y[ind]
            zero_inside, one_inside = is_inside_path(x_center, y_center, patch_size, xy_zero, xy_one)
            if zero_inside and one_inside:
                length_size.append(length_cml)
            elif zero_inside and not one_inside:
                x_exist, y_exist = find_exit_point(xy_zero, xy_one, x_center, y_center, patch_size)
                distance = np.sqrt((x_exist - xy_zero[0]) ** 2 + (y_exist - xy_zero[1]) ** 2)
                length_size.append(distance)
            elif not zero_inside and one_inside:
                x_exist, y_exist = find_exit_point(xy_zero, xy_one, x_center, y_center, patch_size)
                distance = np.sqrt((x_exist - xy_one[0]) ** 2 + (y_exist - xy_one[1]) ** 2)
                length_size.append(distance)
            else:
                point_list = find_entry_exit_point(xy_zero, xy_one, x_center, y_center, patch_size)
                if len(point_list) == 2:
                    distance = np.sqrt((point_list[0][0] - point_list[1][0]) ** 2 +
                                       (point_list[0][1] - point_list[1][1]) ** 2)
                    length_size.append(distance)
                else:
                    raise ValueError("No entry/exit point found within the patch area.")
            if np.abs(sum(length_size) - length_cml) / length_cml > 5:
                raise Exception
            data_array = 0
            for ind, size in zip(index_list, length_size):
                data_array += self.radar_array[:, ind[0], ind[1]] * size / sum(length_size)
            return RadarProjection(data_array=data_array, time_array=self.time_array, lon_lat_zero=lon_lat_zero,
                                   lon_lat_one=lon_lat_one)

    def index_location(self, xy):
        lat_delta = (self.x - xy[0]) ** 2
        lon_delta = (self.y - xy[1]) ** 2
        distance = np.sqrt(lat_delta + lon_delta)
        return np.unravel_index(distance.argmin(), distance.shape)
