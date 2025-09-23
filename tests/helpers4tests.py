import numpy as np
import pynncml as pnc
import pynncml.datasets.sensors_set

MIN_LON_LAT = [57.64367, 11.94063]
MAX_LON_LAT = [57.80246, 12.07351]


def generate_link_set(n_samples, n_links):
    """
    Generate a set of links used in tests only.
    :param n_samples: Number of samples
    :param n_links: Number of links

    """
    link_list = []
    for _ in range(n_links):
        rsl = np.random.rand(n_samples)
        time = np.linspace(0, n_samples - 1, n_samples).astype('int')
        rain = np.zeros(n_samples)

        s = np.random.rand(4)
        lon_zero = MIN_LON_LAT[0] + (MAX_LON_LAT[0] - MIN_LON_LAT[0]) * s[0]
        lat_zero = MIN_LON_LAT[1] + (MAX_LON_LAT[1] - MIN_LON_LAT[1]) * s[1]
        lon_one = MIN_LON_LAT[0] + (MAX_LON_LAT[0] - MIN_LON_LAT[0]) * s[2]
        lat_one = MIN_LON_LAT[1] + (MAX_LON_LAT[1] - MIN_LON_LAT[1]) * s[3]
        l = pnc.datasets.Link(rsl, time, pnc.datasets.MetaData(32, True, 3, 4, 5,
                                                               lon_lat_site_zero=[lon_zero, lat_zero],
                                                               lon_lat_site_one=[lon_one, lat_one]))
        l.add_reference(rain_gauge=rain)
        link_list.append(l)
    return pynncml.datasets.sensors_set.LinkSet(link_list), link_list
