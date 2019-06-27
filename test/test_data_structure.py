import unittest
import torchrain as tr
import torch
import numpy as np


class TestDataStructure(unittest.TestCase):
    n_samples = 100

    def test_link(self):
        rsl = np.random.rand(TestDataStructure.n_samples)
        time = np.linspace(0, 1, TestDataStructure.n_samples)
        rain = np.zeros(TestDataStructure.n_samples)
        l = tr.Link(rsl, rain, time, tr.MetaData(0, 2, 3, 4, 5))
        self.assertTrue(l.start_time() == 0)
        self.assertTrue(l.stop_time() == 1)
        self.assertTrue(l.delta_time() == 1)
        self.assertTrue(not l.has_tsl())
        att = l.attenuation()
        self.assertTrue(np.array_equal(att, -rsl))
        l_min_max = l.create_min_max_link(0.1)
        self.assertTrue(len(l_min_max)==10)

