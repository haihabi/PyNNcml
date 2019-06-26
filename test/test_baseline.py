import unittest
import torchrain as tr
import numpy as np
import torch


class TestBaseLineMethod(unittest.TestCase):
    def test_dynamic(self):
        att = torch.as_tensor(np.random.randn(100))
        dbl = tr.baseline.DynamicBaseLine(8)
        res = dbl(att).numpy()
        att = att.numpy()
        self.assertTrue(len(res) == len(att))
        self.assertTrue(np.min(res) == np.min(att))

    def test_constant(self):
        att = torch.as_tensor(np.random.randn(100))
        cbl = tr.baseline.ConstantBaseLine()
        res = cbl(att, np.zeros(100)).numpy()
        self.assertTrue(len(res) == len(att.numpy()))
        self.assertTrue(np.min(res) == np.min(att.numpy()))
        self.assertTrue(np.sum(att.numpy() - res) == 0)
        res = cbl(att, np.ones(100)).numpy()
        self.assertTrue(np.sum(np.diff(res)) == 0)


if __name__ == '__main__':
    unittest.main()
