import unittest
import numpy as np
import torch
import torchrain as tr


class TestWetDry(unittest.TestCase):
    def test_shape_zero_std(self):
        att = torch.ones(10, 100)
        swd = tr.wet_dry.statistics_wet_dry(0.3, 4)
        res, _ = swd(att)
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_wet_dry_network(self):
        att = torch.ones(1, 100, 4)
        swd = tr.wet_dry.wet_dry_network(1, tr.neural_networks.RNNType.LSTM)
        res, state = swd(att, tr.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == tr.neural_networks.RNN_FEATURES)
        self.assertTrue(state[1].shape[-1] == tr.neural_networks.RNN_FEATURES)
