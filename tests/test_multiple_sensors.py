import unittest
import pynncml as pnc
from tests import helpers4tests as helpers
from tests.test_datasets import OPEN_MRG_TIME_SLICE


class TestMultipeSensors(unittest.TestCase):
    n_samples = 100
    n_link = 20

    def test_infer_multiple(self):
        ls, _ = helpers.generate_link_set(TestMultipeSensors.n_samples, TestMultipeSensors.n_link)
        # ls.plot_links()
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
        idw = pnc.mcm.InverseDistanceWeighting(32, 32)
        map = idw(res, ls)
        self.assertTrue(map.shape[0] == TestMultipeSensors.n_samples)
        self.assertTrue(map.shape[1] == 32)
        self.assertTrue(map.shape[2] == 32)

    def test_idw_real(self):
        link_set, ps = pnc.datasets.load_open_mrg(time_slice=OPEN_MRG_TIME_SLICE, change2min_max=True)

        model = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 8,
                                                                  quantization_delta=0.3)
        imc = pnc.mcm.InferMultipleCMLs(model)
        idw = pnc.mcm.InverseDistanceWeighting(32, 32)
        res = imc(link_set)
        rain_map = idw(res, link_set).numpy()
        self.assertTrue(rain_map.shape[1] == 32)
        self.assertTrue(rain_map.shape[2] == 32)
