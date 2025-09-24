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
from pynncml.datasets.xarray_processing import xarray2link_with_reference, LinkSelection


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


def add_cml_attributes(ds):
    # dictionary of optional and required attributes for variables
    # and coordinates according to OpenSense white paper
    dict_attributes = {
        "time": {
            # "units": "s",    # defining units here interferes with encoding units of time
            "long_name": "time_utc",
            # "_FillValue": np.nan,   # defining units here interferes with encoding
        },
        "cml_id": {
            "long_name": "commercial_microwave_link_identifier",
        },
        "sublink_id": {
            "long_name": "sublink_identifier",
        },
        "site_0_lat": {
            "units": "degrees_in_WGS84_projection",
            "long_name": "site_0_latitude",
        },
        "site_0_lon": {
            "units": "degrees_in_WGS84_projection",
            "long_name": "site_0_longitude",
        },
        "site_0_elev": {
            "units": "meters_above_sea",
            "long_name": "ground_elevation_above_sea_level",
        },
        "site_0_alt": {
            "units": "meters_above_sea",
            "long_name": "antenna_altitude_above_sea_level",
        },
        "site_1_lat": {
            "units": "degrees in WGS84 projection",
            "long_name": "site_1_latitude",
        },
        "site_1_lon": {
            "units": "degrees in WGS84 projection",
            "long_name": "site_1_longitude",
        },
        "site_1_elev": {
            "units": "meters_above_sea",
            "long_name": "ground_elevation_above_sea_level",
        },
        "site_1_alt": {
            "units": "meters_above_sea",
            "long_name": "antenna_altitude_above_sea_level",
        },
        "length": {
            "units": "m",
            "long_name": "distance_between_pair_of_antennas",
        },
        "frequency": {
            "units": "MHz",
            "long_name": "sublink_frequency",
        },
        "polarization": {
            "units": "no units",
            "long_name": "sublink_polarization",
        },
        "tsl": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "transmitted_signal_level",
        },
        "rsl": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "received_signal_level",
        },
        "tsl_max": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "maximum_transmitted_signal_level_over_time_window",
        },
        "tsl_min": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "minimum_transmitted_signal_level_over_time_window",
        },
        "tsl_avg": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "averaged_transmitted_signal_level_over_time_window",
        },
        "tsl_inst": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "instantaneous_transmitted_signal_level",
        },
        "rsl_max": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "maximum_received_signal_level_over_time_window",
        },
        "rsl_min": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "minimum_received_signal_level_over_time_window",
        },
        "rsl_avg": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "averaged_received_signal_level_over_time_window",
        },
        "rsl_inst": {
            "units": "dBm",
            "coordinates": "cml_id, sublink_id, time",
            "long_name": "instantaneous_received_signal_level",
        },
        "temperature_0": {
            "units": "degrees_of_celsius",
            "coordinates": "cml_id, time",
            "long_name": "sensor_temperature_at_site_0",
        },
        "temperature_1": {
            "units": "degrees_of_celsius",
            "coordinates": "cml_id, time",
            "long_name": "sensor_temperature_at_site_1",
        },
    }

    # list of global attributes according to white paper
    global_attr_vars = [
        "title",
        "file author(s)",
        "institution",
        "date",
        "source",
        "history",
        "naming convention",
        "license restrictions",
        "reference",
        "comment",
    ]

    # extract list of variables present in dataset
    ds_vars = list(ds.coords) + list(ds.data_vars)

    # add attributes of variables to dataset
    for v in ds_vars:
        if v in dict_attributes.keys():
            ds[v].attrs = dict_attributes[v]

    # add a placeholder for global attributes that are not given
    for v in global_attr_vars:
        if v not in ds.attrs.keys():
            ds.attrs[v] = "NA"

    # set encoding attributes
    ds.time.encoding['units'] = "seconds since 1970-01-01 00:00:00"

    return ds

def transform_open_mrg(fn, path_to_extract_to,restructure_data=True):
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

    # Add metadata with naming convention as currently used in pycomlink example data file
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
            [df_metadata[df_metadata.Sublink == sublink_id][
                 col_name].values[0] for sublink_id in list(ds.sublink.values)]
        )

    if restructure_data == True:
        # create pandas multiindex for splitting into [cml_id, sublink_id, time]
        df_metadata = df_metadata.reset_index()
        df_metadata = df_metadata.set_index(['Direction', 'Link']).sort_values([('Sublink')], ascending=True)
        ds_multindex = ds.assign_coords({'sublink': df_metadata.index})
        ds_multindex = ds_multindex.unstack()
        ds_multindex = ds_multindex.rename({'Direction': 'sublink_id', 'Link': 'cml_id'})
        ds_multindex['polarization'] = xr.where(ds_multindex['polarization'] == 'Vertical', 'v', 'h')
        ds_multindex['sublink_id'] = xr.where(ds_multindex['sublink_id'] == 'A', 'sublink_1', 'sublink_2')

        # set coordinates that reflect the same properties for both sublinks
        for ds_var_name in ['site_0_lat', 'site_1_lat', 'site_0_lon',
                            'site_1_lon', 'length']:
            ds_multindex = ds_multindex.assign_coords({ds_var_name: (
                'cml_id', ds_multindex.isel(sublink_id=0)[ds_var_name].values)})

        ds_multindex.attrs['comment'] += "\n\nTransformed dataset: \n" \
                                         "In order to meet the Opensense data format conventions the " \
                                         "structure of the dataset was transformed using the code in " \
                                         "opensense_data_downloader_and_transformer.py. This was a joint effort by " \
                                         "Maximilian Graf, Erlend Øydvin, Nico Blettner and Christian Chwala."
        ds_multindex['frequency'] = ds_multindex.frequency * 1000  # to MHz
        ds_multindex['length'] = ds_multindex.length * 1000  # to meter

        ds_multindex = add_cml_attributes(ds_multindex)  # add all attributes afer restructure

        return ds_multindex

    else:
        ds = add_cml_attributes(ds)
        ds.attrs['comment'] += '\nMetadata added with preliminary code from opensense_data_downloader.py'
        return ds

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
    link_set = xarray2link_with_reference(ds, link2gauge_distance, ps, rd, xy_max, xy_min, change2min_max=change2min_max,
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
