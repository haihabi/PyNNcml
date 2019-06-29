import unittest
import torchrain as tr
import torch
import numpy as np


class TestDataStructure(unittest.TestCase):
    n_samples = 100

    def test_link(self):
        rsl = np.random.rand(TestDataStructure.n_samples)
        time = np.linspace(0, TestDataStructure.n_samples - 1, TestDataStructure.n_samples).astype('int')
        rain = np.zeros(TestDataStructure.n_samples)
        l = tr.Link(rsl, rain, time, tr.MetaData(0, 2, 3, 4, 5))
        self.assertTrue(l.start_time().astype('int') == 0)
        self.assertTrue(l.stop_time().astype('int') == 99)
        self.assertTrue(l.delta_time().astype('int') == 99)
        self.assertTrue(not l.has_tsl())
        self.assertTrue(np.array_equal(l.time(), time.astype('datetime64[s]')))
        self.assertEqual(len(l.time()), TestDataStructure.n_samples)
        att = l.attenuation().numpy().flatten()
        self.assertTrue(np.round(np.sum(att + rsl) * 100) == 0)
        l_min_max = l.create_min_max_link(10)
        self.assertTrue(len(l_min_max) == 10)

    def test_min_max_link_generation(self):
        step = 10
        rsl = np.random.rand(TestDataStructure.n_samples)
        time = np.linspace(0, TestDataStructure.n_samples - 1, TestDataStructure.n_samples).astype('int')
        rain = np.zeros(TestDataStructure.n_samples)
        l = tr.Link(rsl, rain, time, tr.MetaData(0, 2, 3, 4, 5))
        res = l.create_min_max_link(step)
        self.assertEqual(l.stop_time(), res.stop_time() + step)
        self.assertEqual(l.start_time(), res.start_time())
        att = res.attenuation()
        self.assertEqual(len(att.shape), 3)
        self.assertEqual(att.shape[2], 2)
