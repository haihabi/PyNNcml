import unittest
import numpy as np
import torch
import torchrain as tr


class TestRainEstimation(unittest.TestCase):
    def test_two_step_constant(self):
        att = torch.ones(10, 100)
        swd = tr.rain_estimation.two_step_constant_baseline(tr.power_law.PowerLawType.MINMAX, 0.3, 6, 0.5)

        res, wd = swd(att, tr.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_two_step_constant_wa_factor(self):
        att = torch.ones(10, 100)
        swd = tr.rain_estimation.two_step_constant_baseline(tr.power_law.PowerLawType.MINMAX, 0.3, 6, 0.5, wa_factor=-1)

        res, wd = swd(att, tr.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_one_step_dynamic(self):
        att = torch.ones(10, 100)
        model = tr.rain_estimation.one_step_dynamic_baseline(tr.power_law.PowerLawType.MINMAX, 0.3, 6)

        res = model(att, tr.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_one_step_network(self):
        att = torch.ones(1, 100, 4)
        swd = tr.rain_estimation.one_step_network(1, tr.neural_networks.RNNType.GRU)
        res, state = swd(att, tr.MetaData(15, 0, 18, 10, 12), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state.shape[-1] == tr.neural_networks.RNN_FEATURES)

    def test_one_step_network_lstm(self):
        att = torch.ones(1, 100, 4)
        swd = tr.rain_estimation.one_step_network(1, tr.neural_networks.RNNType.LSTM)
        res, state = swd(att, tr.MetaData(15, 0, 18, 10, 12), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == tr.neural_networks.RNN_FEATURES)
        self.assertTrue(state[1].shape[-1] == tr.neural_networks.RNN_FEATURES)

    def test_one_step_network_tn_enable(self):
        att = torch.ones(1, 100, 4)
        swd = tr.rain_estimation.one_step_network(1, tr.neural_networks.RNNType.GRU, enable_tn=True, tn_affine=False)
        res, state = swd(att, tr.MetaData(15, 0, 18, 10, 12), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == tr.neural_networks.RNN_FEATURES)
        swd = tr.rain_estimation.one_step_network(1, tr.neural_networks.RNNType.GRU, enable_tn=True, tn_affine=True)
        res, state = swd(att, tr.MetaData(15, 0, 18, 10, 12), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == tr.neural_networks.RNN_FEATURES)

    def test_two_step_network(self):
        att = torch.ones(1, 100, 4)
        swd = tr.rain_estimation.two_step_network(1, tr.neural_networks.RNNType.GRU)
        res, state = swd(att, tr.MetaData(15, 0, 18, 10, 12), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.shape[2] == 2)
        self.assertTrue(state.shape[-1] == tr.neural_networks.RNN_FEATURES)
