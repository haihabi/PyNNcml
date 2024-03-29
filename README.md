[![Build Status](https://travis-ci.com/haihabi/PyNNcml.svg?token=eE741jb2R5GqWJWLJhiE&branch=master)](https://travis-ci.com/haihabi/PyNNcml)
# PyNNcml
A python toolbox based on PyTorch which utilized neural network for rain estimation and classification from commercial microwave link (CMLs) data. This toolbox provides an implementation of algorithms for extracting rain-rate using neural networks and CMLs. Addinaly this project provides an example dataset with data from two CMLs and implementation of performance and robustness metrics.  
# Install
Installation via pip:
```
pip install pynncml
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
In addition, this repository provide PyTorch version of the OpenMRG dataset [9].

# Usage
The following examples:
* Wet Dry Classification using neural network[1] shown in the following [notebook](https://github.com/haihabi/PyNNcml/blob/master/examples/wet_dry_classification_rnn.ipynb) 
* wet Dry Classification using statistic test [6]  shown in the following [notebook](https://github.com/haihabi/PyNNcml/blob/master/examples/wet_dry_classification.ipynb)
* Rain estimation using dynamic baseline[5] shown in the following [notebook](https://github.com/haihabi/PyNNcml/blob/master/examples/rain_estimation_dynamic.ipynb)
* Rain estimation using constant baseline[6] shown in the following [notebook](https://github.com/haihabi/PyNNcml/blob/master/examples/rain_estimation_constant.ipynb)
* Training One Step RNN [4] on the OpenMRG dataset [9] shown in the following [notebook](https://github.com/haihabi/PyNNcml/blob/master/examples/training_rnn.ipynb)



# Model Zoo
In this project we supply a set of trained networks in our [Model Zoo](https://github.com/haihabi/PyNNcml/tree/master/pynncml/model_zoo), this networks are trained on our own dataset which is not publicly available.
The model contains three types of networks: Wet-dry classification network, one-step network (rain estimation only) and two-step network (rain estimation and wet-dry classification). Moreover, we have provided all of these networks with a various number of RNN cells (1, 2, 3). From more details about network structure and results see the publication list.

# Contributing

If you find a bug or have a question, please create a GitHub issue.



# Publications

Please cite one of following paper if you found our neural network model useful. Thanks!

[1] Habi, Hai Victor and Messer, Hagit. "Wet-Dry Classification Using LSTM and Commercial Microwave Links"

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

[2] Habi, Hai Victor and Messer, Hagit. "RNN MODELS FOR RAIN DETECTION"

```
@inproceedings{habi2019rnn,
  title={RNN MODELS FOR RAIN DETECTION},
  author={Habi, Hai Victor and Messer, Hagit},
  booktitle={2019 IEEE International Workshop on Signal Processing Systems  (SiPS)},
  year={2019},
  organization={IEEE}
} 

```

[3] Habi, Hai Victor. "Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links"

```
@article{habi2020,
  title={Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links},
  author={Habi, Hai Victor},
  journal={M.Sc. Thesis, Tel Aviv University},
  year={2019}
}

```

[4] Habi, Hai Victor, and Hagit Messer. "Recurrent neural network for rain estimation using commercial microwave links." IEEE Transactions on Geoscience and Remote Sensing 59.5 (2020): 3672-3681.
```
@article{habi2020recurrent,
  title={Recurrent neural network for rain estimation using commercial microwave links},
  author={Habi, Hai Victor and Messer, Hagit},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={59},
  number={5},
  pages={3672--3681},
  year={2020},
  publisher={IEEE}
}
```

Also, this package contains the implementations of the following papers:

[5] J. Ostrometzky and H. Messer, “Dynamic determination of the baselinelevel in microwave links for rain monitoring from minimum attenuationvalues,”IEEE Journal of Selected Topics in Applied Earth Observationsand Remote Sensing, vol. 11, no. 1, pp. 24–33, Jan 2018.

[6] M. Schleiss and A. Berne, “Identification of dry and rainy periods usingtelecommunication  microwave  links,”IEEE  Geoscience  and  RemoteSensing Letters, vol. 7, no. 3, pp. 611–615, 2010

[7] Jonatan Ostrometzky, Adam Eshel, Pinhas Alpert, and Hagit Messer. Induced bias in attenuation measurements taken from commercial microwave links. In 2017 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3744–3748. IEEE,2017. <br>

[8] Jonatan Ostrometzky, Roi Raich, Adam Eshel, and Hagit Messer.
Calibration of the
attenuation-rain rate power-law parameters using measurements from commercial microwave networks. In 2016 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), pages 3736–3740. IEEE, 2016.

And include PyTorch implementation of the OpenMRG dataset:

[9] van de Beek, Remco CZ, et al. OpenMRG: Open data from Microwave links, Radar, and Gauges for rainfall quantification in Gothenburg, Sweden. No. EGU23-14295. Copernicus Meetings, 2023.


If you found one of those methods usefully please cite.
