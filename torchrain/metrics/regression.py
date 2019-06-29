import numpy as np


def mse(input_array: np.ndarray, reference_array: np.ndarray) -> float:
    return float(np.mean(np.power(input_array - reference_array, 2)))


def nmse(input_array: np.ndarray, reference_array: np.ndarray, epsilon: float = 0.00001) -> float:
    return float(np.mean(np.power(input_array - reference_array, 2) / (epsilon + np.power(reference_array, 2))))
