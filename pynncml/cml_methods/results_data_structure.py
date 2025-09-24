import numpy as np
from dataclasses import dataclass


@dataclass
class CMLResultsDataStructure:
    """
    Attenuation data class
    :param wet_dry_detection: np.ndarray
    :param rain_estimation: np.ndarray
    """
    wet_dry_detection: np.ndarray
    rain_estimation: np.ndarray


    @staticmethod
    def results_types_list():
        return ["wet_dry_detection", "rain_estimation"]