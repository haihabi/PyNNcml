import numpy as np


def accuracy(prediction: np.ndarray, reference: np.ndarray) -> float:
    if len(prediction.shape) == 3:
        prediction = np.argmax(prediction, axis=-1)
    if len(reference.shape) == 3:
        reference = np.argmax(reference, axis=-1)
    if len(reference.shape) != 2 or len(prediction.shape) != 2:
        raise Exception('Input arrays must have 2 or 3 dimension')
    return float(np.mean(prediction == reference))
