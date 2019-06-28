import unittest
import torchrain as tr
import torch
import numpy as np


class TestPowerLaw(unittest.TestCase):
    h_rain_value = 2.2042224
    h_rain_value_min_max = 0.4341563
    r_min = 0.3
    n_samples = 100
    precision = 0.001
    freq = 36

    def test_powerlaw(self):
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLaw(tr.power_law.PowerLawType.ITU, TestPowerLaw.r_min)
        res = pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)).float(), length, TestPowerLaw.freq, 0).numpy()
        self.assertTrue(np.round(100 * (res - TestPowerLaw.h_rain_value))[0] == 0)
        self.assertTrue(len(res) == TestPowerLaw.n_samples)

    def test_powerlaw_min_max(self):
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLaw(tr.power_law.PowerLawType.MINMAX, TestPowerLaw.r_min)
        res = pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)).float(), length, TestPowerLaw.freq, 0).numpy()
        self.assertTrue(np.round(100 * (res - TestPowerLaw.h_rain_value_min_max))[0] == 0)
        self.assertTrue(len(res) == TestPowerLaw.n_samples)

    def test_h_v_diff(self):
        freq = TestPowerLaw.freq + np.random.rand()
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLaw(tr.power_law.PowerLawType.ITU, TestPowerLaw.r_min)

        res_h = pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)).float(), length, freq, 0).numpy()
        res_v = pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)).float(), length, freq, 1).numpy()
        self.assertTrue(np.sum(np.abs(res_h - res_v)) > 0)

    def test_frequency_exception(self):
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLaw(tr.power_law.PowerLawType.ITU, TestPowerLaw.r_min)
        with self.assertRaises(Exception) as context:
            pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)), length, 0.5, 0).numpy()
        self.assertTrue('Frequency must be between 1 Ghz and 100 GHz.' == str(context.exception))

    def test_h_v_exception(self):
        att = 5
        length = 8.3
        pl = tr.power_law.PowerLaw(tr.power_law.PowerLawType.ITU, TestPowerLaw.r_min)
        with self.assertRaises(Exception) as context:
            pl(torch.as_tensor(att * np.ones(TestPowerLaw.n_samples)), length, 3, 'blalba').numpy()
        self.assertTrue('Polarization must be 0 (horizontal) or 1 (vertical).' == str(context.exception))
