# Base on the code from:https://github.com/OpenSenseAction/OPENSENSE_sandbox/blob/main/notebooks/opensense_data_downloader_and_transformer.py
import os
import urllib.request
import zipfile
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from pynncml.datasets.dataset import LinkDataset
from pynncml.datasets.link_data import Link
from pynncml.datasets.gauge_data import PointSensor
from pynncml.datasets import LinkSet, PointSet
from pynncml.datasets.meta_data import MetaData
import numpy as np
from tqdm import tqdm
import time


def download_data_file(url, local_path=".", local_file_name=None, print_output=True):
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


def rain2rain_rate(in_array, window_size=15, step_time=60):
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


def load_open_mrg(data_path="./data/", change2min_max=False, xy_min=None, xy_max=None, time_slice=None,
                  rain_gauge_time_base=900, link2gauge_distance=2000):
    download_open_mrg(local_path=data_path)
    file_location = data_path + "OpenMRG.zip"
    ds = transform_open_mrg(file_location, data_path)

    link_list = []
    if time_slice is not None:
        ds = ds.sel(time=time_slice)

    time_array = ds.time.to_numpy().astype('datetime64[s]')
    ###########################################
    # Process Gauge
    ###########################################

    gauge_metadata = pd.read_csv(os.path.join(data_path, 'gauges/city/CityGauges-metadata.csv'), index_col=0)
    gauge_data = pd.read_csv(os.path.join(data_path, 'gauges/city/CityGauges-2015JJA.csv'), index_col=0)
    time_array_gauge = np.asarray(gauge_data.index).astype("datetime64")
    sel_index = np.logical_and(time_array_gauge >= time_array[0], time_array_gauge <= time_array[-1])
    gauge_list = []
    for g_id in gauge_data.keys():
        gauge_data_array = gauge_data.get(g_id).values[sel_index]
        rain_rate_gauge = rain2rain_rate(gauge_data_array)
        i = np.where(gauge_metadata.index == g_id)[0]
        lon = gauge_metadata.get("Longitude_DecDeg").values[i]
        lat = gauge_metadata.get("Latitude_DecDeg").values[i]
        if not np.any(np.isnan(rain_rate_gauge)):
            ps = PointSensor(rain_rate_gauge, time_array_gauge.astype("int")[sel_index], lat, lon)
            ps = ps.change_time_base(rain_gauge_time_base)
            gauge_list.append(ps)
    ps = PointSet(gauge_list)
    ###########################################
    # Process Links
    ###########################################
    for i in tqdm(range(len(ds.sublink_id))):
        ds_sublink = ds.isel(sublink_id=i)

        md = MetaData(float(ds_sublink.frequency),
                      "Vertical" in str(ds_sublink.polarization),
                      float(ds_sublink.length),
                      None,
                      None,
                      lon_lat_site_zero=[float(ds_sublink.site_0_lon), float(ds_sublink.site_0_lat)],
                      lon_lat_site_one=[float(ds_sublink.site_1_lon), float(ds_sublink.site_1_lat)])
        xy_array = md.xy()
        x_check = xy_min[0] < xy_array[0] and xy_min[0] < xy_array[2] and xy_max[0] > xy_array[2] and xy_max[0] > \
                  xy_array[0]

        y_check = xy_min[1] < xy_array[1] and xy_min[1] < xy_array[3] and xy_max[1] > xy_array[3] and xy_max[1] > \
                  xy_array[1]
        if x_check and y_check:
            rsl = ds_sublink.rsl.to_numpy()
            tsl = ds_sublink.tsl.to_numpy()
            if np.any(np.isnan(rsl)):
                for nan_index in np.where(np.isnan(rsl))[0]:
                    rsl[nan_index] = rsl[nan_index - 1]
            if np.any(np.isnan(tsl)):
                tsl[np.isnan(tsl)] = np.unique(tsl)[0]

            if not np.any(np.isnan(rsl)) and not np.any(np.isnan(tsl)):
                d_min, gauge = ps.find_near_gauge(md.xy_center())
                if d_min < link2gauge_distance:
                    link = Link(rsl,
                                ds_sublink.time.to_numpy().astype('datetime64[s]').astype("int"),
                                meta_data=md,
                                rain_gauge=None,
                                link_tsl=tsl,
                                gauge_ref=gauge)
                    if change2min_max:
                        link = link.create_min_max_link(900)
                    link_list.append(link)
    return LinkSet(link_list), ps


def loader_open_mrg_dataset(data_path="./data/", change2min_max=False, xy_min=None, xy_max=None, time_slice=None,
                            link2gauge_distance=2000):
    link_set, point_set = load_open_mrg(data_path=data_path, change2min_max=change2min_max, xy_min=xy_min,
                                        xy_max=xy_max,
                                        time_slice=time_slice, link2gauge_distance=link2gauge_distance)
    return LinkDataset(link_set)
