import unittest
import pynncml as pnc
import os

OPEN_MRG_TIME_SLICE = slice("2015-06-02", "2015-06-02T02:00:00.000000000")


class TestOpenCML(unittest.TestCase):

    def test_file_exception(self):
        pickle_path = '/bla/bla'
        with self.assertRaises(Exception) as context:
            link_list = pnc.datasets.read_open_cml_dataset(pickle_path)
        self.assertTrue('The input path: ' + pickle_path + ' is not a file' == str(context.exception))

    def test_open_mrg_load(self):
        dataset, _ = pnc.datasets.load_open_mrg(time_slice=OPEN_MRG_TIME_SLICE)
        self.assertTrue(len(dataset) > 0)
