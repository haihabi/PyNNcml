import unittest
import torch
import pynncml as pnc
import os


class TestWetDry(unittest.TestCase):
    def test_shape_zero_std(self):
        att = torch.ones(10, 100)
        swd = pnc.scm.wet_dry.statistics_wet_dry(0.3, 4)
        res, _ = swd(att)
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_wet_dry_network(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.scm.wet_dry.wet_dry_network(1, pnc.neural_networks.RNNType.LSTM)
        res, state = swd(att, pnc.datasets.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == pnc.neural_networks.RNN_FEATURES)
        self.assertTrue(state[1].shape[-1] == pnc.neural_networks.RNN_FEATURES)

