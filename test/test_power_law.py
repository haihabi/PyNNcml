import unittest
import torchrain as tr
import torch
import numpy as np


class TestPowerLaw(unittest.TestCase):
    h_rain_value = 2.2608628
    h_rain_value_min_max = 0.4341563
    r_min = 0.3
    n_samples = 100
    precision = 0.001
    freq = 36

    def test_powerlaw(self):
        pol = 'H'
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLaw(TestPowerLaw.r_min)
        res = pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)), length, TestPowerLaw.freq, pol).numpy()
        self.assertTrue(np.round(100 * (res - TestPowerLaw.h_rain_value))[0] == 0)
        self.assertTrue(len(res) == TestPowerLaw.n_samples)

    def test_powerlaw_min_max(self):
        freq = 36
        pol = 'H'
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLawMinMax(TestPowerLaw.r_min)
        res = pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)).float(), length, TestPowerLaw.freq, pol).numpy()
        self.assertTrue(np.round(100 * (res - TestPowerLaw.h_rain_value_min_max))[0] == 0)
        self.assertTrue(len(res) == TestPowerLaw.n_samples)

    def test_h_v_diff(self):
        freq = TestPowerLaw.freq + np.random.rand()
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLaw(TestPowerLaw.r_min)

        res_h = pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)), length, freq, 'H').numpy()
        res_v = pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)), length, freq, 'V').numpy()
        self.assertTrue(np.sum(np.abs(res_h - res_v)) > 0)

    def test_frequency_exception(self):
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLaw(TestPowerLaw.r_min)
        with self.assertRaises(Exception) as context:
            pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)), length, 0.5, 'H').numpy()
        self.assertTrue('Frequency must be between 1 Ghz and 100 GHz.' == str(context.exception))

    def test_h_v_exception(self):
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLaw(TestPowerLaw.r_min)
        with self.assertRaises(Exception) as context:
            pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)), length, 3, 'blalba').numpy()
        self.assertTrue('Polarization must be V, v, H or h.' == str(context.exception))


if __name__ == '__main__':
    unittest.main()
