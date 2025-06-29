import unittest
import pynncml as pnc

import torch

from pynncml.datasets import LinkSelection

OPEN_MRG_TIME_SLICE = slice("2015-06-02", "2015-06-02T02:00:00.000000000")


class TestOpenCML(unittest.TestCase):

    def test_open_mrg_load(self):
        dataset, _, _ = pnc.datasets.load_open_mrg(time_slice=OPEN_MRG_TIME_SLICE)
        self.assertTrue(len(dataset) > 0)

    def test_open_mrg_load_dataset(self):
        dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=OPEN_MRG_TIME_SLICE)
        data_loader = torch.utils.data.DataLoader(dataset, 16)
        for rain, rsl, tsl, metadata in data_loader:
            self.assertTrue(rain.shape[1] == rsl.shape[1])
            self.assertTrue(tsl.shape[1] == rsl.shape[1])
            self.assertTrue(tsl.shape[2] == rsl.shape[2])
            self.assertTrue(rsl.shape[0] == 16)
            self.assertTrue(tsl.shape[0] == 16)
            self.assertTrue(metadata.shape[0] == 16)
            break

    # def test_open_mrg_load_dataset_multiple_gauges(self):
    #     dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=OPEN_MRG_TIME_SLICE, multiple_gauges_per_link=True)
    #     data_loader = torch.utils.data.DataLoader(dataset, 16)
    #     for rain, rsl, tsl, metadata in data_loader:
    #         self.assertTrue(rain.shape[1] == rsl.shape[1])
    #         self.assertTrue(tsl.shape[1] == rsl.shape[1])
    #         self.assertTrue(tsl.shape[2] == rsl.shape[2])
    #         self.assertTrue(rsl.shape[0] == 16)
    #         self.assertTrue(tsl.shape[0] == 16)
    #         self.assertTrue(metadata.shape[0] == 16)
    #         break

    def test_open_mrg_load_dataset_radar_only(self):
        dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=OPEN_MRG_TIME_SLICE,
                                                       link_selection=LinkSelection.RADARONLY)
        data_loader = torch.utils.data.DataLoader(dataset, 16)
        for rain, rsl, tsl, metadata in data_loader:
            self.assertTrue(rain.shape[1] == rsl.shape[1])
            self.assertTrue(tsl.shape[1] == rsl.shape[1])
            self.assertTrue(tsl.shape[2] == rsl.shape[2])
            self.assertTrue(rsl.shape[0] == 16)
            self.assertTrue(tsl.shape[0] == 16)
            self.assertTrue(metadata.shape[0] == 16)
            break

    def test_open_mrg_load_dataset_radar_all(self):
        dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=OPEN_MRG_TIME_SLICE, link_selection=LinkSelection.ALL)
        data_loader = torch.utils.data.DataLoader(dataset, 16)
        for rain, rsl, tsl, metadata in data_loader:
            self.assertTrue(rain.shape[1] == rsl.shape[1])
            self.assertTrue(tsl.shape[1] == rsl.shape[1])
            self.assertTrue(tsl.shape[2] == rsl.shape[2])
            self.assertTrue(rsl.shape[0] == 16)
            self.assertTrue(tsl.shape[0] == 16)
            self.assertTrue(metadata.shape[0] == 16)
            break

    # def test_open_mrg_load_dataset_multiple_all(self):
    #     dataset = pnc.datasets.loader_open_mrg_dataset(time_slice=OPEN_MRG_TIME_SLICE, link_selection=LinkSelection.ALL,
    #                                                    multiple_gauges_per_link=True)
    #     data_loader = torch.utils.data.DataLoader(dataset, 16)
    #     for rain, rsl, tsl, metadata in data_loader:
    #         self.assertTrue(rain.shape[1] == rsl.shape[1])
    #         self.assertTrue(tsl.shape[1] == rsl.shape[1])
    #         self.assertTrue(tsl.shape[2] == rsl.shape[2])
    #         self.assertTrue(rsl.shape[0] == 16)
    #         self.assertTrue(tsl.shape[0] == 16)
    #         self.assertTrue(metadata.shape[0] == 16)
    #         break
