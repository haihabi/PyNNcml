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
 
* Rain estimation using dynamic baseline []
* Rain estimation using constant baseline []
* Wet Dry Classification using statistic test []

# Model Zoo
In this project we supply a set of trained networks in our [Model Zoo](https://github.com/haihabi/torch_rain/blob/master/model_zoo/model_zoo.md), this networks are trained on our own dataset which is not publicly available.
The model contains three types of networks: Wet-dry classification network, one-step network (rain estimation only) and two-step network (rain estimation and wet-dry classification). Moreover, we have provided all of these networks with a various number of RNN cells (1, 2, 3). From more details about network structure and results see the publication list.


# Publication

Please cite one of following paper if you found our model useful. Thanks!

> Habi, Hai Victor and Messer, Hagit. "Wet-Dry Classification Using LSTM and Commercial Microwave Links"

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
