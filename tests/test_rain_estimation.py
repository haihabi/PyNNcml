import unittest
import numpy as np
import torch
import pynncml as pnc


class TestRainEstimation(unittest.TestCase):
    n_samples = 100

    def test_two_step_constant(self):
        att = torch.ones(10, 100)
        swd = pnc.rain_estimation.two_step_constant_baseline(pnc.power_law.PowerLawType.MAX, 0.3, 6, 0.5)

        res, wd = swd(att, pnc.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_two_step_constant_wa_factor(self):
        att = torch.ones(10, 100)
        swd = pnc.rain_estimation.two_step_constant_baseline(pnc.power_law.PowerLawType.MAX, 0.3, 6, 0.5, wa_factor=-1)

        res, wd = swd(att, pnc.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_one_step_dynamic(self):
        att = torch.ones(10, 100)
        model = pnc.rain_estimation.one_step_dynamic_baseline(pnc.power_law.PowerLawType.MAX, 0.3, 6)

        res = model(att, pnc.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_one_step_network(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.rain_estimation.one_step_network(1, pnc.neural_networks.RNNType.GRU)
        res, state = swd(att, pnc.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state.shape[-1] == pnc.neural_networks.RNN_FEATURES)

    def test_one_step_network_lstm(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.rain_estimation.one_step_network(1, pnc.neural_networks.RNNType.LSTM)
        res, state = swd(att, pnc.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == pnc.neural_networks.RNN_FEATURES)
        self.assertTrue(state[1].shape[-1] == pnc.neural_networks.RNN_FEATURES)

    def test_one_step_network_tn_enable(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.rain_estimation.one_step_network(1, pnc.neural_networks.RNNType.GRU, enable_tn=True, tn_affine=False)
        res, state = swd(att, pnc.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == pnc.neural_networks.RNN_FEATURES)
        swd = pnc.rain_estimation.one_step_network(1, pnc.neural_networks.RNNType.GRU, enable_tn=True, tn_affine=True)
        res, state = swd(att, pnc.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == pnc.neural_networks.RNN_FEATURES)

    def test_two_step_network(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.rain_estimation.two_step_network(1, pnc.neural_networks.RNNType.GRU)
        res, state = swd(att, pnc.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.shape[2] == 2)
        self.assertTrue(state.shape[-1] == pnc.neural_networks.RNN_FEATURES)

    def test_backbone_exception(self):
        pickle_path = '/bla/bla'
        with self.assertRaises(Exception) as context:
            swd = pnc.rain_estimation.two_step_network(1, 3)
        self.assertTrue('Unknown RNN type' == str(context.exception))

    def test_one_step_min_max(self):
        step = 10
        rsl = np.random.rand(TestRainEstimation.n_samples)
        time = np.linspace(0, TestRainEstimation.n_samples - 1, TestRainEstimation.n_samples).astype('int')
        rain = np.zeros(TestRainEstimation.n_samples)
        l = pnc.Link(rsl, rain, time, pnc.MetaData(0, 2, 3, 4, 5))
        l_min_max = l.create_min_max_link(5)
        model = pnc.rain_estimation.one_step_dynamic_baseline(pnc.power_law.PowerLawType.MAX, 0.3, 6)
        res = model(l_min_max.attenuation(), pnc.MetaData(15, 0, 18, 10, 12))
