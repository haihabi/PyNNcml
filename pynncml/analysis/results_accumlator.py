import numpy as np


class ResultsAccumulator:
    def __init__(self):
        self.results_dict = dict()

    def clear(self):
        self.results_dict.clear()

    def add_results(self, **kwargs):
        for k, v in kwargs.items():
            if self.results_dict.get(k) is None:
                self.results_dict.update({k: []})
            self.results_dict[k].append(v)

    def get_results(self, name):
        return np.asarray(self.results_dict[name])


class AverageMetric(ResultsAccumulator):
    def get_results(self, name):
        return np.mean(super().get_results(name))



