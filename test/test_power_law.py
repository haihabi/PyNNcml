import unittest
import torchrain as tr
import torch


class TestPowerLaw(unittest.TestCase):
    def test_something(self):
        freq = 36
        pol = 'H'
        att = 5
        length = 8.3
        pl = tr.pl.PowerLaw(0.3)
        res = pl(torch.as_tensor(att), length, freq, pol)


if __name__ == '__main__':
    unittest.main()
