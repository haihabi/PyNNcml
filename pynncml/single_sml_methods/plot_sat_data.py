import numpy as np
from matplotlib import pyplot as plt, dates as mdates


def PlotRainVsSatData(rain_array,rain_timestamp, sat_array,sat_timestamp):
    fig, ax = plt.subplots(1, figsize=(10, 8))
    # Plots of rain estimation from Sat signal
    ax1 = ax.twinx()

    ax.plot_date(rain_timestamp, sat_array, '-', color='purple', label='Esno Data ')
    ax1.plot_date(sat_timestamp, rain_array, '-', label="Rain Gauge")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # ax1.set_ylim([0, 10])
    # ax.set_ylim([-15, 12])
    ax.set_ylabel('Satellite  Signal Es/No [dB]')

    # ax1.set_ylim([0, 80])
    ax1.set_ylabel('Rain[mm/h]')

    ax.legend(loc='upper left')
    ax1.legend(loc='upper right')
    ax.grid()

    fig.suptitle(' Rain Gauge Vs Satellite Signal')
    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def PlotRainEstimationVsGaugeData(RainEst, RainDF, SatDF, ST, FT):
    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax1 = ax.twinx()

    ax.plot_date(SatDF.timestamp, SatDF.esno, '-', color='purple', label='Esno Data ')
    ax.plot_date(SatDF.timestamp, ST, '-', color='c', label='SlowTracker')
    ax.plot_date(SatDF.timestamp, FT, '-', color='b', label='FastTracker')
    ax1.plot_date(SatDF.timestamp, RainEst, '-', color='r', label='Rain Rate')
    ax1.plot_date(SatDF.timestamp, np.cumsum(RainEst / 120), '-', color='k', label='Cumulative Sum Rain Rate')

    ax1.plot_date(RainDF.datetime, RainDF.value, '-', color='g', label="Rain Gauge [HaKfar HaYarok]")
    ax1.plot_date(RainDF.datetime, np.cumsum(RainDF.value / 12), '-', color='y',
                  label="Cumulative Rain Gauge [HaKfar HaYarok]")
    RainGaugSum = np.sum(RainDF.value / 12)
    EstRainSum = np.sum(RainEst / 120)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # ax1.set_ylim([0, 10])
    ax.set_ylim([-15, 12])
    ax.set_ylabel('Satellite  Signal Es/No [dB]')

    ax1.set_ylim([0, 80])
    ax1.set_ylabel('Rain[mm/h]')

    ax.legend(loc='upper left')
    ax1.legend(loc='upper right')
    ax.grid()

    fig.suptitle('satellite link measurements at ' + str(
        RainDF.datetime.iloc[100].date()) + "\n \n" + "Sum HAKFAR HAYAROK   " + str(RainGaugSum) + "   [mm]" +

                 "      |        Sum Algorithm   " + str(EstRainSum) + "   [mm]" + " \n")
    # fig.suptitle(' Rain Gauge Vs Sat Data at Kfar Saba ' + str(RainDF.datetime.iloc[100].date()))
    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()
