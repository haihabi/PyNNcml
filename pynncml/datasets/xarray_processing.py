def xarray_time_slice(ds, start_time, end_time):
    return ds.sel(time=slice(start_time, end_time))


def xarray_location_slice(ds, lon_min, lon_max, lat_min, lat_max):
    return ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
