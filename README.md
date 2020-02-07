[![Build Status](https://travis-ci.com/haihabi/torch_rain.svg?token=eE741jb2R5GqWJWLJhiE&branch=master)](https://travis-ci.com/haihabi/torch_rain)
# Torch Rain
Library for rain estimation and detection built with pytorch. 
This library provide an implementations of algorithms for extracting rain-rate using data from commercial microwave links (CMLs). Addinaly this project provide an example dataset with data from two CMLs and implementation of perfomance and robustness metrics.  

# Install
The install via pip
```
pip instal torchrain

```


# Projects Structure

1. Wet Dry Classification
2. Baseline 
3. Power Law 
4. Rain estimation
5. Metrics
6. Robustness
# Dataset
This repository includes an example of a dataset with a reference rain gauge.
# Examples
Example of
* Wet Dry Classification using neural network[1] shown in the following [notebook](../examples/wet_dry_classification_rnn.ipynb) 
* wet Dry Classification using statistic test [5]  shown in the following [notebook](../examples/wet_dry_classification.ipynb)
* Rain estimation using dynamic baseline[4] shown in the following [notebook](../examples/rain_estimation_dynamic.ipynb)
* Rain estimation using constant baseline[5] shown in the following [notebook](../examples/rain_estimation_constant.ipynb)


# Model Zoo
In this project we supply a set of trained networks in our [Model Zoo](https://github.com/haihabi/torch_rain/blob/master/model_zoo/model_zoo.md), this networks are trained on our own dataset which is not publicly available.
The model contains three types of networks: Wet-dry classification network, one-step network (rain estimation only) and two-step network (rain estimation and wet-dry classification). Moreover, we have provided all of these networks with a various number of RNN cells (1, 2, 3). From more details about network structure and results see the publication list.


# Publications

Please cite one of following paper if you found our neural network model useful. Thanks!

>[1] Habi, Hai Victor and Messer, Hagit. "Wet-Dry Classification Using LSTM and Commercial Microwave Links"

```
@inproceedings{habi2018wet,
  title={Wet-Dry Classification Using LSTM and Commercial Microwave Links},
  author={Habi, Hai Victor and Messer, Hagit},
  booktitle={2018 IEEE 10th Sensor Array and Multichannel Signal Processing Workshop (SAM)},
  pages={149--153},
  year={2018},
  organization={IEEE}
} 

```

>[2] Habi, Hai Victor and Messer, Hagit. "RNN MODELS FOR RAIN DETECTION"

```
@inproceedings{habi2019rnn,
  title={RNN MODELS FOR RAIN DETECTION},
  author={Habi, Hai Victor and Messer, Hagit},
  booktitle={2019 IEEE International Workshop on Signal Processing Systems  (SiPS)},
  year={2019},
  organization={IEEE}
} 

```

>[3] Habi, Hai Victor. "Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links"

```
@article{habi2020,
  title={Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links},
  author={Habi, Hai Victor},
  journal={M.Sc. Thesis, Tel Aviv University},
  year={2019}
}

```

Also this package contains the implementaion of the following papers:

[4] J. Ostrometzky and H. Messer, “Dynamic determination of the baselinelevel in microwave links for rain monitoring from minimum attenuationvalues,”IEEE Journal of Selected Topics in Applied Earth Observationsand Remote Sensing, vol. 11, no. 1, pp. 24–33, Jan 2018.

[5] M. Schleiss and A. Berne, “Identification of dry and rainy periods usingtelecommunication  microwave  links,”IEEE  Geoscience  and  RemoteSensing Letters, vol. 7, no. 3, pp. 611–615, 2010

If you found on of this methods usefully please cite.