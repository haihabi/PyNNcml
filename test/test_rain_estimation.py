import unittest
import numpy as np
import torch
import torchrain as tr


class TestRainEstimation(unittest.TestCase):
    def test_shape_zero_std(self):
        att = torch.ones(10, 100)
        swd = tr.rain_estimation.two_step_constant_baseline(tr.power_law.PowerLawType.MINMAX, 0.3, 6, 0.5)

        res, wd = swd(att, tr.MetaData(15, 'H', 18, 10, 12))
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)
