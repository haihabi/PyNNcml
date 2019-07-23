import unittest
import os
from torchrain.model_common import get_model_from_zoo, ModelType, RNNType


class TestDownload(unittest.TestCase):
    def test_something(self):
        for model in [ModelType.ONESTEP, ModelType.TWOSTEP, ModelType.WETDRY]:
            for rnn_type in [RNNType.LSTM, RNNType.GRU]:
                for n_layers in [1, 2, 3]:
                    file = get_model_from_zoo(model, rnn_type, n_layers)
                    print(file)
                    self.assertTrue(os.path.isfile(file))
