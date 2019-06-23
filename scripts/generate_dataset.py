import pandas as pd
import glob
import os
import numpy as np
import time
import datetime
import torchrain as tr


def change2min_base(data_array, time_array, step_size=60):
    start_time = np.min(time_array)
    stop_time = np.max(time_array)

    start_time = start_time + 3600 - start_time % 3600
    stop_time = stop_time - stop_time % 3600

    low_time = np.linspace(start_time, stop_time - step_size,
                           np.ceil((stop_time - start_time) / step_size).astype('int'))
    high_time = np.linspace(start_time + step_size, stop_time,
                            np.ceil((stop_time - start_time) / step_size).astype('int'))
    time_vector = []
    data_vector = []
    for lt, ht in zip(low_time, high_time):  # loop over high and low time step
        data = data_array[(time_array >= lt) * (time_array < ht)]
        if len(data) > 0:
            data_vector.append(data[0])
        else:
            data_vector.append(0)
        time_vector.append(ht)  # append time step
    return np.asarray(data_vector), np.asarray(time_vector)


def rain_depth2rain_rate(gauge_array, window_size=30):
    res = np.zeros(gauge_array.shape[0])
    scale = np.zeros(gauge_array.shape[0])
    start = False

    for i in reversed(range(gauge_array.shape[0])):
        if gauge_array[i] == 0.0:
            if start and (index - i) >= window_size:
                v = gauge_array[index]
                res[(i + 1):(index + 1)] = v * (3600 / 60)
                scale[(i + 1):(index + 1)] = 1 / len(res[(i + 1):(index + 1)])
                start = False
        else:
            if start:
                v = gauge_array[index]
                res[(i + 1):(index + 1)] = v * (3600 / 60)
                scale[(i + 1):(index + 1)] = 1 / len(res[(i + 1):(index + 1)])
            index = i
            start = True
    res = res * scale
    return np.convolve(res, np.ones(15) * (1 / 15), mode='same')


def _time_date2unix(date_input, time_input):
    if '-' in date_input:
        return time.mktime(datetime.datetime.strptime(date_input + ' ' + time_input, "%d-%m-%Y %H:%M").timetuple())
    else:
        return time.mktime(datetime.datetime.strptime(date_input + ' ' + time_input, "%d/%m/%Y %H:%M").timetuple())


def read_links_data(input_base_path):
    files = glob.glob(input_base_path + "*.txt")
    data_base = None
    time_base = None
    for file in files:
        data_new = pd.read_csv(file, header=-1)

        data = data_new[1].values
        time = data_new[0].values
        time, index = np.unique(time.astype(np.datetime64).astype('uint64'), return_index=True)
        data = data[index]
        if data_base is None:
            data_base = data
            time_base = time
        else:
            data_base = np.concatenate([data_base, data])
            time_base = np.concatenate([time_base, time])
    time_base, index = np.unique(60 * np.floor(time_base / 60), return_index=True)
    data_base = data_base[index]
    time_base = time_base.astype('int')
    t_diff = np.diff(time_base)

    start_array = np.concatenate([np.asarray(-1).reshape(1), np.where(t_diff > 60)[0]])
    end_array = np.concatenate([np.where(t_diff > 60)[0], np.asarray(-1).reshape(1)])
    data_list = []
    for start, end in zip(start_array, end_array):
        t = time_base[(start + 1):end]
        d = data_base[(start + 1):end]
        data_list.append((d, t))
    return data_list


base_path = os.path.join(os.path.expanduser("~"), "projects/open_dataset/pelephon_2007-2008/AirPort_City-Ben_Guryon/")
link_a = read_links_data(base_path)
base_path = os.path.join(os.path.expanduser("~"), "projects/open_dataset/pelephon_2007-2008/Revadim-Sorek/")
link_b = read_links_data(base_path)
base_path = os.path.join(os.path.expanduser("~"), "Downloads/drive-download-20190621T163247Z-001/")
link_data = {'air_port': link_a, 'other': link_b}

files = glob.glob(base_path + "*.csv")
name = 'air_port'
for file in files:
    if name in file:
        rain_data_new = pd.read_csv(file, header=0)
        time_array = np.asarray([_time_date2unix(d, t) for d, t in zip(rain_data_new['date'], rain_data_new['time'])])
        time_array, index = np.unique(time_array, return_index=True)
        rain_array = rain_data_new['rain'].values[index]
        rain_array, time_array = change2min_base(rain_array, time_array)
        rain_array = rain_depth2rain_rate(rain_array)
        # Math the correct link sequnce
