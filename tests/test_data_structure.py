import unittest
import pynncml as pnc
import torch
import numpy as np


class TestDataStructure(unittest.TestCase):
    n_samples = 100

    def test_handle_excpetion(self):
        att = torch.ones(100)

        with self.assertRaises(Exception) as context:
            swd = pnc.scm.rain_estimation.two_step_constant_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 6, 0.5)
            res, wd = swd(att, pnc.datasets.MetaData(15, 0, 18, 10, 12))
        self.assertTrue(
            'The input attenuation vector dont match min max format or regular format' == str(context.exception))

    def test_link(self):
        rsl = np.random.rand(TestDataStructure.n_samples)
        time = np.linspace(0, TestDataStructure.n_samples - 1, TestDataStructure.n_samples).astype('int')
        rain = np.zeros(TestDataStructure.n_samples)
        l = pnc.datasets.Link(rsl, rain, time, pnc.datasets.MetaData(0, 2, 3, 4, 5))
        l.plot()
        self.assertEqual(l.meta_data.as_tensor().shape[1], 5)
        self.assertEqual(l.meta_data.as_tensor().shape[0], 1)
        self.assertEqual(len(l.rain()), TestDataStructure.n_samples)
        self.assertEqual(len(l.cumulative_rain()), TestDataStructure.n_samples)
        self.assertEqual(l.step(), np.diff(time).min() / 3600)

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
        l = pnc.datasets.Link(rsl, rain, time, pnc.datasets.MetaData(0, 2, 3, 4, 5))
        res = l.create_min_max_link(step)
        self.assertEqual(l.stop_time(), res.stop_time() + step)
        self.assertEqual(l.start_time(), res.start_time())
        res.plot()
        att = res.attenuation()
        self.assertEqual(len(att.shape), 2)
        self.assertEqual(att.shape[0], 2)
        t = res.as_tensor(5)

        self.assertEqual(len(t.shape), 2)
        self.assertEqual(t.shape[1], 4)

    def test_link_with_tsl(self):
        rsl = np.random.rand(TestDataStructure.n_samples)
        tsl = np.random.rand(TestDataStructure.n_samples)
        time = np.linspace(0, TestDataStructure.n_samples - 1, TestDataStructure.n_samples).astype('int')
        rain = np.zeros(TestDataStructure.n_samples)
        l = pnc.datasets.Link(rsl, rain, time, pnc.datasets.MetaData(0, 2, 3, 4, 5), link_tsl=tsl)
        self.assertTrue(l.start_time().astype('int') == 0)
        self.assertTrue(l.stop_time().astype('int') == 99)
        self.assertTrue(l.delta_time().astype('int') == 99)
        self.assertTrue(l.has_tsl())
        self.assertTrue(np.array_equal(l.time(), time.astype('datetime64[s]')))
        self.assertEqual(len(l.time()), TestDataStructure.n_samples)
        att = l.attenuation().numpy().flatten()
        self.assertTrue(np.round(np.sum(att + tsl - rsl) * 100) == 0)
        l_min_max = l.create_min_max_link(10)
        self.assertTrue(len(l_min_max) == 10)
        self.assertEqual(len(l_min_max.attenuation().shape), 2)
