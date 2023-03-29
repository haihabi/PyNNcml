import unittest
import pynncml as pnc
import os


class TestOpenCML(unittest.TestCase):

    # @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this tests on Travis CI.")
    # def test_load(self):
    #     link_list = pnc.read_open_cml_dataset('../dataset/open_cml.p')
    #     [self.assertTrue(isinstance(d, pnc.Link)) for d in link_list]
    #     self.assertTrue(len(link_list) == 20)

    def test_file_exception(self):
        pickle_path = '/bla/bla'
        with self.assertRaises(Exception) as context:
            link_list = pnc.datasets.read_open_cml_dataset(pickle_path)
        self.assertTrue('The input path: ' + pickle_path + ' is not a file' == str(context.exception))

    def test_open_mrg_load(self):
        import pynncml as pnc

        dataset = pnc.datasets.load_open_mrg()
        dataset.plot_links()
        pass
