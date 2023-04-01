# Base on the code from:https://github.com/OpenSenseAction/OPENSENSE_sandbox/blob/main/notebooks/opensense_data_downloader_and_transformer.py
import os
import urllib.request
import zipfile
from functools import partial
import pandas as pd
import xarray as xr
from pynncml.datasets.link_data import Link
from pynncml.datasets import LinkSet
from pynncml.datasets.meta_data import MetaData
import numpy as np
from tqdm import tqdm
import time


# TODO: Change to constants

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


def load_open_mrg(data_path="./data/", change2min_max=True, xy_min=None, xy_max=None, time_slice=None):
    download_open_mrg(local_path=data_path)
    file_location = data_path + "OpenMRG.zip"
    ds = transform_open_mrg(file_location, data_path)

    link_list = []
    if time_slice is not None:
        ds = ds.sel(time=time_slice)
    for i in tqdm(range(len(ds.sublink_id))):
        ds_sublink = ds.isel(sublink_id=i)

        md = MetaData(float(ds_sublink.frequency),
                      "Vertical" in str(ds_sublink.polarization),
                      float(ds_sublink.length),
                      None,
                      None,
                      lon_lat_site_zero=[float(ds_sublink.site_0_lat), float(ds_sublink.site_0_lon)],
                      lon_lat_site_one=[float(ds_sublink.site_1_lat), float(ds_sublink.site_1_lon)])
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
                link = Link(rsl,
                            ds_sublink.time.to_numpy().astype('datetime64[s]').astype("int"),
                            meta_data=md,
                            rain_gauge=None,  # TODO: add rain gauge measurements
                            link_tsl=tsl)
                if change2min_max:
                    link = link.create_min_max_link(900)
                link_list.append(link)
    return LinkSet(link_list)
