import unittest

import matplotlib.pyplot as plt

import pynncml as pnc
import numpy as np
from tests import helpers4tests as helpers


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
        # xy_min = [0.57e6, 1.32e6]
        # xy_max = [0.5875e6, 1.335e6]
        time_slice = slice("2015-06-01", "2015-06-2")
        link_set = pnc.datasets.load_open_mrg(time_slice=time_slice)
        print(link_set.area())

        model = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.MAX, 0.3, 8,
                                                                  quantization_delta=0.3)
        imc = pnc.mcm.InferMultipleCMLs(model)
        res = imc(link_set)
        idw = pnc.mcm.InverseDistanceWeighting(32, 32)
        rain_map = idw(res, link_set).numpy()
        print("a")
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots()

        ims = []
        for i in range(rain_map.shape[0]):
            # x += np.pi / 15
            # y += np.pi / 30
            im = ax.imshow(rain_map[i, :, :], animated=True)
            if i == 0:
                ax.imshow(rain_map[i, :, :])  # show an initial one first
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)
        plt.show()
