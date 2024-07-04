import unittest
import numpy as np
from pynncml.simulation.generate_rain_field import get_rain_filed_generation_function


class TestRainFieldGenerationFunction(unittest.TestCase):
    def test_rain_field_generation(self):
        # Define the height and width for the rain field
        height, width = 32, 32
        # Get the rain field generation function
        sample_rain_field = get_rain_filed_generation_function(height, width)

        # Define parameters for the rain field generation
        rain_coverage = 0.5
        n_peaks = 3
        peak_rain_rate = 100
        batch_size = 1

        # Generate the rain field
        rain_field = sample_rain_field(rain_coverage, n_peaks, peak_rain_rate, batch_size)

        # Check if the generated rain field has the correct shape
        self.assertEqual(rain_field.shape, (batch_size, height, width))

        # Check if the values in the generated rain field are within the expected range
        self.assertTrue(np.all(rain_field >= 0))
        self.assertTrue(np.all(rain_field <= peak_rain_rate))


if __name__ == '__main__':
    unittest.main()
