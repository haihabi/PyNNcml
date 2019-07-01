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
