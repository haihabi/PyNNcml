import unittest
import pynncml as pnc
import torch
import numpy as np


class TestTimeNormalization(unittest.TestCase):
    batch_size = 2

    def test_indecator(self):
        tn = pnc.neural_networks.TimeNormalization(0.5, 10)
        state = tn.init_state('cpu', TestTimeNormalization.batch_size)
        data = torch.as_tensor(np.random.randn(TestTimeNormalization.batch_size, 100, 10)).float()
        # ind = torch.as_tensor(np.round(np.random.rand(TestTimeNormalization.batch_size, 100)).astype('int')).float()
        n_data, new_state = tn(data, state)
        self.assertEqual(n_data.shape[0], data.shape[0])
        self.assertEqual(n_data.shape[1], data.shape[1])
        self.assertEqual(n_data.shape[2], data.shape[2])
