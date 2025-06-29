from enum import Enum

import numpy as np
from tqdm import tqdm

from pynncml.datasets import MetaData, Link, LinkSet


def xarray_time_slice(ds, start_time, end_time):
    """
    Slice the xarray dataset based on time
    :param ds: xarray dataset
    :param start_time: start time
    :param end_time: end time
    :return: xarray dataset
    """
    return ds.sel(time=slice(start_time, end_time))


def xarray_location_slice(ds, lon_min, lon_max, lat_min, lat_max):
    """
    Slice the xarray dataset based on location
    :param ds: xarray dataset
    :param lon_min: min longitude
    :param lon_max: max longitude
    :param lat_min: min latitude
    :param lat_max: max latitude
    """
    return ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))


def xarray_sublink2link(ds_sublink, gauge=None, radar_cml_projection=None):
    """
    Convert xarray sublink to link
    :param ds_sublink: xarray dataset
    :param gauge: gauge data
    :return: Link
    """
    md = MetaData(float(ds_sublink.frequency),
                  "Vertical" in str(ds_sublink.polarization),
                  float(ds_sublink.length),
                  None,
                  None,
                  lon_lat_site_zero=[float(ds_sublink.site_0_lon), float(ds_sublink.site_0_lat)],
                  lon_lat_site_one=[float(ds_sublink.site_1_lon), float(ds_sublink.site_1_lat)])
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
                    rain_gauge=None,
                    link_tsl=tsl,
                    gauge_ref=gauge,
                    radar_cml_projection_ref=radar_cml_projection)
    else:
        link = None
    return link


class LinkSelection(Enum):
    """
    Enum for link selection
    """
    ALL = 0
    GAUGEONLY = 1
    RADARONLY = 2

    def enable_gauge(self):
        """
        Check if gauge links are enabled
        :return: True if gauge links are enabled, False otherwise
        """
        return self == LinkSelection.ALL or self == LinkSelection.GAUGEONLY

    def enable_radar(self):
        """
        Check if radar links are enabled
        :return: True if radar links are enabled, False otherwise
        """
        return self == LinkSelection.ALL or self == LinkSelection.RADARONLY

def xarray2link(ds,
                link2gauge_distance,
                ps,
                rd=None,
                xy_max=None,
                xy_min=None,
                change2min_max=False,
                min_max_window: int = 900,
                multiple_gauges_per_link=False,
                link_selection=LinkSelection.ALL):
    """
    Convert xarray dataset to link set
    :param ds: xarray dataset
    :param link2gauge_distance: distance between the link and the gauge
    :param ps: PointSet
    :param rd: RadarData
    :param xy_max: max x and y
    :param xy_min: min x and y
    :param change2min_max: change to min max
    :param min_max_window: window size for min max
    :param multiple_gauges_per_link: if True, multiple gauges can be assigned to the same link
    :param link_selection: LinkSelection enum to select the type of links to return
    :return: LinkSet
    """
    link_list = []
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
        if xy_min is None or xy_max is None:
            is_in_area = True
        else:
            x_check = xy_min[0] < xy_array[0] and xy_min[0] < xy_array[2] and xy_max[0] > xy_array[2] and xy_max[0] > \
                      xy_array[0]

            y_check = xy_min[1] < xy_array[1] and xy_min[1] < xy_array[3] and xy_max[1] > xy_array[3] and xy_max[1] > \
                      xy_array[1]
            is_in_area = x_check and y_check

        if is_in_area:
            if ps is None:
                link = xarray_sublink2link(ds_sublink)
            else:
                if rd is not None and link_selection.enable_radar():
                    radar_cml_projection = rd.radar_projection2cml(md.lon_lat_site_zero, md.lon_lat_site_one)
                    active_link = True
                else:
                    radar_cml_projection = None
                    active_link = False

                if multiple_gauges_per_link:
                    d_array,gauge = ps.find_near_gauges(md.xy_center(), link2gauge_distance)
                    active_link = active_link or (len(gauge) > 0 and link_selection.enable_gauge())
                    if len(gauge)==0:
                        gauge = None

                else:
                    d_min, gauge = ps.find_near_gauge(md.xy_center())
                    active_link = active_link or (d_min < link2gauge_distance and link_selection.enable_gauge())

                if active_link:
                    link = xarray_sublink2link(ds_sublink, gauge, radar_cml_projection)
                else:
                    link = None  # Link is too far from the gauge

            if change2min_max and link is not None:
                link = link.create_min_max_link(min_max_window)
            if link is not None: link_list.append(link)
    return LinkSet(link_list)
