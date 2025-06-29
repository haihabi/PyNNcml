# Base on the code from:https://github.com/OpenSenseAction/OPENSENSE_sandbox/blob/main/notebooks/opensense_data_downloader_and_transformer.py
import os
import urllib.request
import zipfile
from functools import partial

import pandas as pd
import xarray as xr

from pynncml.datasets.dataset import LinkDataset
from pynncml.datasets.gauge_data import PointSensor
from pynncml.datasets import PointSet
import numpy as np

from pynncml.datasets.radar_data import RadarData
from pynncml.datasets.xarray_processing import xarray2link, LinkSelection


def download_data_file(url, local_path=".", local_file_name=None, print_output=True):
    """
    Download a file from a URL to a local path
    :param url: URL to download from
    :param local_path: Local path to download to
    :param local_file_name: Local file name to save as
    :param print_output: Print download information
    """
    if not os.path.exists(local_path):
        if print_output:
            print(f"Creating path {local_path}")
        os.makedirs(local_path)

    if local_file_name is None:
        local_file_name = url.split("/")[-1]

    if os.path.exists(os.path.join(local_path, local_file_name)):
        print(
            f"File already exists at desired location {os.path.join(local_path, local_file_name)}"
        )
        print("Not downloading!")
        return

    if print_output:
        print(f"Downloading {url}")
        print(f"to {local_path}/{local_file_name}")

    request_return_meassage = urllib.request.urlretrieve(
        url, os.path.join(local_path, local_file_name)
    )
    return request_return_meassage


download_open_mrg = partial(
    download_data_file,
    url="https://zenodo.org/record/7107689/files/OpenMRG.zip",
)


def transform_open_mrg(fn, path_to_extract_to):
    """
    Transform the OpenMRG dataset to a xarray dataset
    :param fn: File name
    :param path_to_extract_to: Path to extract to
    """
    # For this ZIP file we cannot extract only the CML dataset since
    # the NetCDF with the CML dataset is quite large. This seems to
    # lead to crashes when reding directly from the ZIP file via Python.
    with zipfile.ZipFile(fn) as zfile:
        zfile.extractall(path_to_extract_to)

    # Read metadata and dataset
    df_metadata = pd.read_csv(os.path.join(path_to_extract_to, 'cml/cml_metadata.csv'), index_col=0)
    ds = xr.open_dataset(os.path.join(path_to_extract_to, 'cml/cml.nc'))

    # Add metadata with naming convention as currently used in pycomlink example dataset file
    for col_name, ds_var_name in [
        ('NearLatitude_DecDeg', 'site_0_lat'),
        ('NearLongitude_DecDeg', 'site_0_lon'),
        ('FarLatitude_DecDeg', 'site_1_lat'),
        ('FarLongitude_DecDeg', 'site_1_lon'),
        ('Frequency_GHz', 'frequency'),
        ('Polarization', 'polarization'),
        ('Length_km', 'length'),
    ]:
        ds.coords[ds_var_name] = (
            ('sublink'),
            [df_metadata[df_metadata.Sublink == sublink_id][col_name].values[0] for sublink_id in
             list(ds.sublink.values)]
        )

    ds.attrs['comment'] += '\nMetadata added with preliminary code from opensense_data_downloader.py'

    # Change "sublink" to "sublink_id"
    ds = ds.rename({"sublink": "sublink_id"})

    return ds


def rain2rain_rate(in_array: np.ndarray, window_size: int = 15, step_time: int = 60):
    """
    Convert rain to rain rate
    :param in_array: Input array
    :param window_size: Window size
    :param step_time: Step time
    """
    res = np.zeros(in_array.shape[0])
    scale = np.zeros(in_array.shape[0])
    start = False

    for i in reversed(range(in_array.shape[0])):
        if in_array[i] == 0.0:
            if start and (index - i) >= window_size:
                v = in_array[index]
                res[(i + 1):(index + 1)] = v * (3600 / step_time)
                scale[(i + 1):(index + 1)] = 1 / len(res[(i + 1):(index + 1)])
                start = False
        else:
            if start:
                v = in_array[index]
                res[(i + 1):(index + 1)] = v * (3600 / step_time)
                scale[(i + 1):(index + 1)] = 1 / len(res[(i + 1):(index + 1)])
            index = i
            start = True
    res = res * scale
    return np.convolve(res, np.ones(window_size) * (1 / window_size), mode='same')


def radar2rain(dbz_tensor):
    """
    Convert radar reflectivity (dBZ) to rain rate using the Z-R relationship.
    :param dbz_tensor: Input tensor of radar reflectivity in dBZ
    :return: Tensor of rain rate in mm/h
    """
    gain = 0.4
    offset = -30
    # dbz_tensor = radar_tensor * gain + offset
    radar_rain_tensor = np.power(10, ((dbz_tensor / 10) - np.log10(200)) * (1 / 1.5))
    radar_rain_tensor[dbz_tensor < 5] = 0
    radar_rain_tensor[np.round((dbz_tensor - offset) / gain) == 255] = 0
    radar_rain_tensor = np.nan_to_num(radar_rain_tensor, nan=0)
    return radar_rain_tensor


def load_open_mrg(data_path="./data/", change2min_max=False, xy_min=None, xy_max=None, time_slice=None,
                  rain_gauge_time_base=900, link2gauge_distance=2000, window_size_in_min=15,
                  multiple_gauges_per_link=False,
                  link_selection: LinkSelection = LinkSelection.GAUGEONLY):
    """
    Load OpenMRG dataset and process it to create a LinkSet and PointSet.
    :param data_path: Path to store the dataset
    :param change2min_max: Change to min max dataset
    :param xy_min: Minimum xy use to filter the dataset based on xy location
    :param xy_max: Maximum xy use to filter the dataset based on xy location
    :param time_slice: Time slice to filter the dataset
    :param rain_gauge_time_base: Time base for the rain gauge data in seconds
    :param link2gauge_distance: Link to gauge distance in meter
    :param window_size_in_min: Window size in minute for rain rate calculation
    :param multiple_gauges_per_link: Use multiple gauges per link
    :param link_selection: Link selection strategy
    :return: LinkSet, PointSet and RadarData
    """
    download_open_mrg(local_path=data_path)
    file_location = data_path + "OpenMRG.zip"
    ds = transform_open_mrg(file_location, data_path)

    if time_slice is not None:
        ds = ds.sel(time=time_slice)

    time_array = ds.time.to_numpy().astype('datetime64[s]')
    ###########################################
    # Process Gauge
    ###########################################
    gauge_metadata = pd.read_csv(os.path.join(data_path, 'gauges/city/CityGauges-metadata.csv'), index_col=0)
    gauge_data = pd.read_csv(os.path.join(data_path, 'gauges/city/CityGauges-2015JJA.csv'), index_col=0)
    time_array_gauge = np.asarray([np.datetime64(i[:-1]) for i in gauge_data.index.to_numpy()])
    sel_index = np.logical_and(time_array_gauge >= time_array[0], time_array_gauge <= time_array[-1])
    gauge_list = []
    for g_id in gauge_data.keys():
        gauge_data_array = gauge_data.get(g_id).values[sel_index]
        rain_rate_gauge = rain2rain_rate(gauge_data_array, window_size=window_size_in_min)
        i = np.where(gauge_metadata.index == g_id)[0]
        lon = gauge_metadata.get("Longitude_DecDeg").values[i]
        lat = gauge_metadata.get("Latitude_DecDeg").values[i]
        if not np.any(np.isnan(rain_rate_gauge)):
            ps = PointSensor(rain_rate_gauge, time_array_gauge.astype("int")[sel_index], lon, lat)
            ps = ps.change_time_base(rain_gauge_time_base)
            gauge_list.append(ps)
    ps = PointSet(gauge_list)
    ###########################################
    # Process Radar
    ###########################################
    radar_file = os.path.join(data_path, 'radar/radar.nc')
    if not os.path.exists(radar_file):
        raise FileNotFoundError("Radar file not found. Please check the data path.")
    radar_ds = xr.open_dataset(radar_file)
    if time_slice is not None:
        radar_ds = radar_ds.sel(time=time_slice)
    time = np.asarray(radar_ds.time)
    lat = np.asarray(radar_ds.lat)
    lon_array = np.asarray(radar_ds.lon)
    radar_array = radar2rain(np.asarray(radar_ds.data))
    rd = RadarData(radar_array, lat, lon_array, time.astype('datetime64[s]').astype("int"))
    rd = rd.change_time_base(rain_gauge_time_base)
    ###########################################
    # Process Links
    ###########################################
    link_set = xarray2link(ds, link2gauge_distance, ps, rd, xy_max, xy_min, change2min_max=change2min_max,
                           multiple_gauges_per_link=multiple_gauges_per_link, link_selection=link_selection)
    return link_set, ps, rd


def loader_open_mrg_dataset(data_path="./data/",
                            change2min_max=False,
                            xy_min=None,
                            xy_max=None,
                            time_slice=None,
                            link2gauge_distance=2000,
                            window_size_in_min=15,
                            multiple_gauges_per_link=False,
                            link_selection: LinkSelection = LinkSelection.GAUGEONLY):
    """
    Load OpenMRG dataset
    :param data_path: Path to store the dataset
    :param change2min_max: Change to min max dataset
    :param xy_min: Minimum xy use to filter the dataset based on xy location
    :param xy_max: Maximum xy use to filter the dataset based on xy location
    :param time_slice: Time slice to filter the dataset
    :param link2gauge_distance: Link to gauge distance in meter
    :param window_size_in_min: Window size in minute
    :param multiple_gauges_per_link: Use multiple gauges per link
    :param link_selection: Link selection strategy
    :return: LinkDataset
    """
    link_set, point_set, _ = load_open_mrg(data_path=data_path, change2min_max=change2min_max, xy_min=xy_min,
                                           xy_max=xy_max,
                                           time_slice=time_slice, link2gauge_distance=link2gauge_distance,
                                           window_size_in_min=window_size_in_min,
                                           multiple_gauges_per_link=multiple_gauges_per_link,
                                           link_selection=link_selection)
    return LinkDataset(link_set, point_set)
