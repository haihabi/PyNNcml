import unittest
import torch
import pynncml as pnc
import os


class TestWetDry(unittest.TestCase):
    def test_shape_zero_std(self):
        att = torch.ones(10, 100)
        swd = pnc.scm.wet_dry.statistics_wet_dry(0.3, 4)
        res, _ = swd(att)
        self.assertTrue(res.shape[0] == 10)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(res.numpy().sum() == 0)

    def test_wet_dry_network(self):
        att = torch.ones(1, 100, 4)
        swd = pnc.scm.wet_dry.wet_dry_network(1, pnc.neural_networks.RNNType.LSTM)
        res, state = swd(att, pnc.MetaData(15, 0, 18, 10, 12).as_tensor(), swd.init_state(batch_size=1))
        self.assertTrue(res.shape[0] == 1)
        self.assertTrue(res.shape[1] == 100)
        self.assertTrue(state[0].shape[-1] == pnc.neural_networks.RNN_FEATURES)
        self.assertTrue(state[1].shape[-1] == pnc.neural_networks.RNN_FEATURES)

    # @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this tests on Travis CI.")
    # def test_with_real_data(self):
    #     open_cml_dataset = pnc.read_open_cml_dataset('../dataset/open_cml.p')  # read OpenCML dataset
    #
    #     link_index = 0
    #     link_data = open_cml_dataset[link_index]  # select a link
    #     link_min_max = link_data.create_min_max_link(900)
    #
    #     rnn = pnc.scm.wet_dry.wet_dry_network(1, pnc.neural_networks.RNNType.GRU)
    #     wd_classification, _ = rnn(link_min_max.as_tensor(constant_tsl=5), link_data.meta_data.as_tensor(),
    #                                rnn.init_state())  # run classification method
    #     self.assertTrue(wd_classification.shape[0] == 1)
    #     self.assertTrue(wd_classification.shape[1] == len(link_min_max))
    #     self.assertTrue(wd_classification.shape[2] == 1)
