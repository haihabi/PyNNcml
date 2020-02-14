import unittest
import pynncml as pnc
import numpy as np
import torch


class TestBaseLineMethod(unittest.TestCase):
    def test_dynamic(self):
        att = torch.as_tensor(np.random.randn(1, 100))
        dbl = pnc.baseline.DynamicBaseLine(8)
        res = dbl(att).numpy()
        att = att.numpy()
        self.assertTrue(len(res) == len(att))
        self.assertTrue(np.min(res) == np.min(att))

    def test_one_step_dynamic_exception(self):
        att = torch.as_tensor(np.random.randn(100))
        with self.assertRaises(Exception) as context:
            dbl = pnc.baseline.DynamicBaseLine(8)
            dbl(att)
        self.assertEqual('Dynamic base line module only accepts inputs with shape length equal to 2',
                         str(context.exception))

    def test_constant(self):
        att = torch.as_tensor(np.random.randn(1, 100))
        cbl = pnc.baseline.ConstantBaseLine()
        res = cbl(att, np.zeros([1, 100])).numpy()
        self.assertTrue(len(res) == len(att.numpy()))
        self.assertTrue(np.min(res) == np.min(att.numpy()))
        self.assertTrue(np.sum(att.numpy() - res) == 0)
        res = cbl(att, np.ones([1, 100])).numpy()
        self.assertTrue(np.sum(np.diff(res)) == 0)
