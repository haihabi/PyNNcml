
# Examples of using the PyNNCML library

This document provides an overview of the examples on how to use the PyNNCML library to obtain various whether monotoning information using CMLs data.
# The examples are divided into two main categories:
## Example Of Single CML

Here, we provide examples of how to use the PyNNCML library to obtain various whether monotoning information using CMLs data. Those method apply on a single CML data and provide the following examples and tasks:
1. Rain Detection
2. Rain Estimation
3. Training RNN Model on OpenMRG Dataset

| Task            | Algorithm                           | Notebooks                                             | Description                                                  | 
|-----------------|-------------------------------------|-------------------------------------------------------|--------------------------------------------------------------|
| Rain Detection  | Classification using Rnn [1,2,3]    | [Notebook](wet_dry_classification_rnn.ipynb) | This notebook  run rnn model for rain detection              |
| Rain Detection  | Classification using Std Window [6] | [Notebook](wet_dry_classification.ipynb)     | This notebook run std rolloing window for rain detection     |
| Rain Estimation | Constant Baseline  [6]              | [Notebook](rain_estimation_constant.ipynb)   | This notebook run rain estimation using constant baseline.   |
| Rain Estimation | Dynamic Baseline    [5,7,8]         | [Notebook](rain_estimation_dynamic.ipynb)    | This notebook run rain estimation using dynamic baseline.    |
| Rain Estimation | Direct RNN Estimation [4,3]         | [Notebook](rain_estimation_rnn.ipynb)        | This notebook run rain estimation using RNN Model.           |
| Rain Estimation | RNN Training Example [1,2,3,4]      | [Notebook](tutorials/data_driven_tutorial.ipynb)               | This notebook train an RNN model on the OpenMRG Dataset [10] |


## Example Of Multiple CML and Rain Filed Mapping.


| Task                     | Algorithm                       | Notebooks                                          | Description                                                                                    | 
|--------------------------|---------------------------------|----------------------------------------------------|------------------------------------------------------------------------------------------------|
| Rain Field Generation    | Rain Field generation using GAN | [Notebook](rain_generator_notebook.ipynb) | This notebook  run RainGAN to generate rain field                                              |
| Rain Field Interpolation | Interpolation Using IDW         | [Notebook](rain_map_interpolation.ipynb)  | This notebook dynamic baseline followed by IDW interpolation                                   |
| Rain Field Interpolation | Interpolation Using GMZ  [9]    | [Notebook](rain_map_interpolation.ipynb)  | This notebook dynamic baseline followed by GMZ preprocessing and then interpolation using IDW. |

## Example Of Single SML.
| Task            | Algorithm                           | Notebooks                             | Description                                                  |
|-----------------|-------------------------------------|---------------------------------------|--------------------------------------------------------------|
| Rain Estimation| Kalman base rain estimation         | [Notebook](rain_estimation_sml.ipynb) | This notebook run Kalman filter for rain estimation using SML data. |

## References

[1] Habi, Hai Victor and Messer, Hagit. "Wet-Dry Classification Using LSTM and Commercial Microwave Links"



[2] Habi, Hai Victor and Messer, Hagit. "RNN MODELS FOR RAIN DETECTION"



[3] Habi, Hai Victor. "Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links"


[4] Habi, Hai Victor, and Hagit Messer. "Recurrent neural network for rain estimation using commercial microwave links." IEEE Transactions on Geoscience and Remote Sensing 59.5 (2020): 3672-3681.


[5] J. Ostrometzky and H. Messer, “Dynamic determination of the baselinelevel in microwave links for rain monitoring from minimum attenuationvalues,”IEEE Journal of Selected Topics in Applied Earth Observationsand Remote Sensing, vol. 11, no. 1, pp. 24–33, Jan 2018.

[6] M. Schleiss and A. Berne, “Identification of dry and rainy periods using telecommunication  microwave  links,”IEEE  Geoscience  and  RemoteSensing Letters, vol. 7, no. 3, pp. 611–615, 2010

[7] Jonatan Ostrometzky, Adam Eshel, Pinhas Alpert, and Hagit Messer. Induced bias in attenuation measurements taken from commercial microwave links. In 2017 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3744–3748. IEEE,2017. <br>

[8] Jonatan Ostrometzky, Roi Raich, Adam Eshel, and Hagit Messer.
Calibration of the
attenuation-rain rate power-law parameters using measurements from commercial microwave networks. In 2016 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), pages 3736–3740. IEEE, 2016.

[9] Goldshtein, Oren, Hagit Messer, and Artem Zinevich. "Rain rate estimation using measurements from commercial telecommunications links." IEEE Transactions on signal processing 57.4 (2009): 1616-1625.

[10] van de Beek, Remco CZ, et al. OpenMRG: Open data from Microwave links, Radar, and Gauges for rainfall quantification in Gothenburg, Sweden. No. EGU23-14295. Copernicus Meetings, 2023.

