import unittest
import numpy as np
import torch
import pynncml as pnc


class TestRainEstimation(unittest.TestCase):
    n_samples = 100

    def test_two_step_constant(self):
        att = torch.ones(10, 100)
        swd = pnc.scm.rain_estimation.two_step_constant_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 6, 0.5)

        res, wd, _ = swd(att, pnc.datasets.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_two_step_constant_wa_factor(self):
        att = torch.ones(10, 100)
        swd = pnc.scm.rain_estimation.two_step_constant_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 6, 0.5,
                                                                 wa_factor=-1)

        res, wd, _ = swd(att, pnc.datasets.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_one_step_dynamic(self):
        att = torch.ones(10, 100)
        model = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 6,
                                                                  quantization_delta=0.1)

        res, _ = model(att, pnc.datasets.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_one_step_network(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.scm.rain_estimation.one_step_network(1, pnc.neural_networks.DNNType.GRU)
        res, state = swd(att, pnc.datasets.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state.shape[-1] == pnc.neural_networks.RNN_FEATURES)

    def test_one_step_network_lstm(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.scm.rain_estimation.one_step_network(1, pnc.neural_networks.DNNType.LSTM)
        res, state = swd(att, pnc.datasets.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == pnc.neural_networks.RNN_FEATURES)
        self.assertTrue(state[1].shape[-1] == pnc.neural_networks.RNN_FEATURES)

    def test_one_step_network_tn_enable(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.scm.rain_estimation.one_step_network(1, pnc.neural_networks.DNNType.GRU, enable_tn=True)
        res, state = swd(att, pnc.datasets.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == pnc.neural_networks.RNN_FEATURES)
        swd = pnc.scm.rain_estimation.one_step_network(1, pnc.neural_networks.DNNType.GRU, enable_tn=True)
        res, state = swd(att, pnc.datasets.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == pnc.neural_networks.RNN_FEATURES)

    def test_two_step_network(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.scm.rain_estimation.two_step_network(1, pnc.neural_networks.DNNType.GRU)
        res, state = swd(att, pnc.datasets.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.shape[2] == 2)
        self.assertTrue(state.shape[-1] == pnc.neural_networks.RNN_FEATURES)

    def test_backbone_exception(self):
        with self.assertRaises(Exception) as context:
            swd = pnc.scm.rain_estimation.two_step_network(1, 3)
        self.assertTrue('Unknown RNN type' == str(context.exception))

    def test_one_step_min_max(self):
        rsl = np.random.rand(TestRainEstimation.n_samples)
        time = np.linspace(0, TestRainEstimation.n_samples - 1, TestRainEstimation.n_samples).astype('int')
        rain = np.zeros(TestRainEstimation.n_samples)
        l = pnc.datasets.Link(rsl, time, pnc.datasets.MetaData(0, 2, 3, 4, 5))
        l.add_reference(rain_gauge=rain)
        l_min_max = l.create_min_max_link(5)
        model = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 6,
                                                                  quantization_delta=1.0)
        res = model(l_min_max.attenuation(), pnc.datasets.MetaData(15, 0, 18, 10, 12))
