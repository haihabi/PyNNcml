import unittest
import pynncml as pnc
import os


class TestOpenCML(unittest.TestCase):

    def test_file_exception(self):
        pickle_path = '/bla/bla'
        with self.assertRaises(Exception) as context:
            link_list = pnc.datasets.read_open_cml_dataset(pickle_path)
        self.assertTrue('The input path: ' + pickle_path + ' is not a file' == str(context.exception))

    def test_open_mrg_load(self):
        xy_min = [0.57e6, 1.32e6]
        xy_max = [0.5875e6, 1.335e6]
        time_slice = slice("2015-06-01", "2015-06-10")
        dataset = pnc.datasets.load_open_mrg(xy_min=xy_min, xy_max=xy_max, time_slice=time_slice)
        print("a")
