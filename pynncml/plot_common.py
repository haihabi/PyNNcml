from matplotlib import pyplot as plt
import matplotlib.dates as md
import numpy as np


def change_x_axis_time_format(input_format: str):
    """
    Change the x-axis time format.
    :param input_format: The input format.
    """
    ax = plt.gca()
    xfmt = md.DateFormatter(input_format)
    ax.xaxis.set_major_formatter(xfmt)


def plot_wet_dry_detection_mark(in_ax, in_x, in_detection_array, in_rain_array):
    """


    """
    rain_max = np.max(in_rain_array)
    in_ax.fill_between(in_x, rain_max, where=np.logical_or(np.logical_and(in_detection_array == 1, in_rain_array > 0),
                                                        np.logical_and(in_detection_array == 0, in_rain_array == 0)),
                       facecolor='green', alpha=.5, label="Detection")
    in_ax.fill_between(in_x, rain_max, where=np.logical_and(in_detection_array == 0, in_rain_array > 0), facecolor='red',
                       alpha=.5, label="Mis-Detection")
    in_ax.fill_between(in_x, rain_max, where=np.logical_and(in_detection_array == 1, in_rain_array == 0), facecolor='blue',
                       alpha=.5, label="False Alarm")
