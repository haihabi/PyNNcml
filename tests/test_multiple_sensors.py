import unittest
import pynncml as pnc
from tests import helpers4tests as helpers
from tests.test_datasets import OPEN_MRG_TIME_SLICE


class TestMultipeSensors(unittest.TestCase):
    n_samples = 100
    n_link = 20

    def test_infer_multiple(self):
        ls, _ = helpers.generate_link_set(TestMultipeSensors.n_samples, TestMultipeSensors.n_link)
        model = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 6,
                                                                  quantization_delta=1.0)
        imc = pnc.mcm.InferMultipleCMLs(model)
        res = imc(ls)
        self.assertTrue(res.shape[0] == 20)
        self.assertTrue(res.shape[-1] == 100)

    def test_idw(self):
        ls, _ = helpers.generate_link_set(TestMultipeSensors.n_samples, TestMultipeSensors.n_link)
        model = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 6,
                                                                  quantization_delta=1.0)
        imc = pnc.mcm.InferMultipleCMLs(model)
        res = imc(ls)
        idw = pnc.mcm.generate_link_set_idw(ls)
        map = idw(res)
        self.assertTrue(map.shape[0] == TestMultipeSensors.n_samples)

    def test_idw_real(self):
        link_set, ps = pnc.datasets.load_open_mrg(time_slice=OPEN_MRG_TIME_SLICE, change2min_max=True)

        model = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 8,
                                                                  quantization_delta=0.3)
        imc = pnc.mcm.InferMultipleCMLs(model)
        idw = pnc.mcm.generate_link_set_idw(link_set)
        res = imc(link_set)
        rain_map = idw(res).numpy()
        self.assertTrue(rain_map.shape[1] == 48)
        self.assertTrue(rain_map.shape[2] == 16)

    def test_gmz_real(self):
        link_set, ps = pnc.datasets.load_open_mrg(time_slice=OPEN_MRG_TIME_SLICE, change2min_max=True)

        model = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 8,
                                                                  quantization_delta=0.3)
        imc = pnc.mcm.InferMultipleCMLs(model)
        gmz = pnc.mcm.generate_link_set_gmz(link_set)
        res = imc(link_set)
        rain_map, _ = gmz(res)
        rain_map = rain_map.numpy()
        self.assertTrue(rain_map.shape[1] == 48)
        self.assertTrue(rain_map.shape[2] == 16)
