import unittest
import torchrain as tr
import numpy as np
import torch


class TestBaseLineMethod(unittest.TestCase):
    def test_dynamic(self):
        att = torch.as_tensor(np.random.randn(100))
        dbl = tr.bl.DynamicBaseLine(8)
        res = dbl(att)


if __name__ == '__main__':
    unittest.main()
