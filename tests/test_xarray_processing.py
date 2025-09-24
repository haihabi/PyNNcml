import unittest

import poligrain as plg

from pynncml.apis.xarray_processing.wet_dry_methods import create_wet_dry_std


class TestOpenCML(unittest.TestCase):

    def test_poligrain_to_xarray(self):
        (ds_rad,
         ds_cmls,
         ds_gauges_municp,
         ds_gauge_smhi) = plg.example_data.load_openmrg(data_dir="example_data", subset="8d")
        nn_base=create_wet_dry_std()
        nn_base(ds_cmls)

