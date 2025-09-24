import numpy as np


def accuracy(prediction: np.ndarray, reference: np.ndarray) -> float:
    r"""
    The accuracy function compute the top1 accuracy of predication array.
        .. math::
            acc=\frac{1}{N}\sum_i^N 1(p_i=r_i)

    where acc is the accuracy measurement, p is the predication array, r is the reference array and 1 is the indicator function.

    :param prediction: A numpy array of shape :math:`[N_b,N_s]` or :math:`[N_b,N_s,N_c]` where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence and :math:`N_c` is the number of class
    :param reference: A numpy array of shape :math:`[N_b,N_s]` or :math:`[N_b,N_s,N_c]` where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence and :math:`N_c` is the number of class
    :return: a floating point number that represent the accuracy measurement
    """
    if len(prediction.shape) == 3:
        prediction = np.argmax(prediction, axis=-1)
    if len(reference.shape) == 3:
        reference = np.argmax(reference, axis=-1)
    if len(reference.shape) != 2 or len(prediction.shape) != 2:
        raise Exception('Input arrays must have 2 or 3 dimension')
    return float(np.mean(prediction == reference))
