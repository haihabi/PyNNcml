from matplotlib import pyplot as plt
import matplotlib.dates as md


def change_x_axis_time_format(input_format: str):
    ax = plt.gca()
    xfmt = md.DateFormatter(input_format)
    ax.xaxis.set_major_formatter(xfmt)
