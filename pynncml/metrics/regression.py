import numpy as np


def mse(input_array: np.ndarray, reference_array: np.ndarray) -> float:
    r"""
    The mse function compute the mean square error of predication array.
        .. math::
            mse=\frac{1}{N}\sum_i^N (p_i-r_i)^2

    where mse is the mean square error measurement, p is the predication array, r is the reference array.
    Note:reference array shape must be equal to input array shape

    :param input_array: A numpy array of any shape
    :param reference_array: A numpy array of any shape
    :return: a floating point number that represent the mean square error measurement
    """
    return float(np.mean(np.power(input_array - reference_array, 2)))


def nmse(input_array: np.ndarray, reference_array: np.ndarray, epsilon: float = 0.00001) -> float:
    r"""
    The nmse function compute the normalized mean square error of predication array.
        .. math::
            nmse=\frac{1}{N}\sum_i^N \frac{(p_i-r_i)^2}{r_i^2+\epsilon}

    where nmse is the normalized mean square error measurement, p is the predication array, r is the reference array
    and epsilon is a floating point number fo numeric stability.
    Note:reference array shape must be equal to input array shape

    :param input_array: A numpy array of any shape
    :param reference_array: A numpy array of any shape
    :param epsilon: a floating point number fo numric stabiliy
    :return: a floating point number that represent the normalized mean square error measurement
    """
    return float(np.mean(np.power(input_array - reference_array, 2) / (epsilon + np.power(reference_array, 2))))


def rmse(input_array: np.ndarray, reference_array: np.ndarray) -> float:
    r"""
    The rmse function compute the mean square error of predication array.
        .. math::
            mse=\sqrt{\frac{1}{N}\sum_i^N (p_i-r_i)^2}

    where mse is the mean square error measurement, p is the predication array, r is the reference array.
    Note:reference array shape must be equal to input array shape

    :param input_array: A numpy array of any shape
    :param reference_array: A numpy array of any shape
    :return: a floating point number that represent the mean square error measurement
    """
    return float(np.sqrt(np.mean(np.power(input_array - reference_array, 2))))
