import unittest
import os
import torch
from pynncml.model_zoo.model_common import get_model_from_zoo, ModelType, DNNType
from pynncml.single_cml_methods.rain_estimation import two_step_network
from pynncml.single_cml_methods.rain_estimation import one_step_network
from pynncml.single_cml_methods.wet_dry import wet_dry_network

def build_model(in_model_type,in_n_layers,in_rnn_type):
    if in_model_type == ModelType.ONESTEP:
        return one_step_network(in_n_layers,in_rnn_type)
    elif in_model_type == ModelType.TWOSTEP:
        return two_step_network(in_n_layers,in_rnn_type)
    elif in_model_type == ModelType.WETDRY:
        return wet_dry_network(in_n_layers,in_rnn_type)



class TestModelZoo(unittest.TestCase):
    def test_get_model(self):
        for model in [ModelType.ONESTEP, ModelType.TWOSTEP, ModelType.WETDRY]:
            for rnn_type in [DNNType.LSTM, DNNType.GRU]:
                for n_layers in [1, 2, 3]:
                    file = get_model_from_zoo(model, rnn_type, n_layers)
                    print(file)
                    _model=build_model(model,n_layers,rnn_type)
                    self.assertTrue(os.path.isfile(file))

    def test_unknown_model(self):
        with self.assertRaises(Exception) as context:
            get_model_from_zoo(10, DNNType.GRU, 1)
        self.assertTrue('unknown model:' + str(10) == str(context.exception))

    def test_unknown_rnn(self):
        with self.assertRaises(Exception) as context:
            get_model_from_zoo(ModelType.WETDRY, 3, 1)
        self.assertTrue('unknown RNN type:' + str(3) == str(context.exception))

    def test_unknown_layer(self):
        with self.assertRaises(Exception) as context:
            get_model_from_zoo(ModelType.WETDRY, DNNType.GRU, 12)
        self.assertTrue('there is not model with:' + str(12) + 'layers' == str(context.exception))
