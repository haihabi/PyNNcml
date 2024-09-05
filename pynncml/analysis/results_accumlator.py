import numpy as np
from prettytable import PrettyTable


class ResultsAccumulator:
    def __init__(self):
        """
        Initialize the results accumulator

        """
        self.results_dict = dict()

    def clear(self):
        """
        Clear the results accumulator
        """
        self.results_dict.clear()

    def add_results(self, **kwargs):
        """
        Add results to the accumulator
        :param kwargs: key-value pairs of the results
        """
        for k, v in kwargs.items():
            if self.results_dict.get(k) is None:
                self.results_dict.update({k: []})
            self.results_dict[k].append(v)

    def get_results(self, name: str):
        """
        Get the results of a specific metric
        """
        return np.asarray(self.results_dict[name])


class AverageMetric(ResultsAccumulator):
    def get_results(self, name: str):
        """
        Get the average of the results
        """
        return np.mean(super().get_results(name))


class GroupAnalysis:
    def __init__(self):
        """
        Initialize the group analysis
        """
        self.reference = []
        self.estimation = []

    def append(self, reference: np.ndarray, estimation: np.ndarray):
        """
        Append the reference and estimation data
        """
        self.reference.append(reference)
        self.estimation.append(estimation)

    def _analysis(self, ref, est, group_selection, normalized=False):
        """
        Perform the analysis
        """
        ref = np.concatenate(self.reference)
        est = np.concatenate(self.estimation)
        bias = []
        rmse = []
        group_data = []
        for gs, ge in group_selection:
            sel = np.logical_and(gs <= ref, ref < ge)
            _ref = ref[sel]
            _est = est[sel]
            group_data.append((_ref, _est))
            _delta = _ref - _est
            _rmse = np.sqrt(np.mean(_delta ** 2))
            _bias = np.mean(_delta)
            if normalized:
                ref_mean = np.mean(_ref)
                _rmse /= ref_mean
                _bias /= ref_mean
            bias.append(_bias)
            rmse.append(_rmse)
        return rmse, bias, group_selection, group_data

    def run_analysis(self, group_selection):
        """
        Run the analysis
        """
        ref = np.concatenate(self.reference)
        est = np.concatenate(self.estimation)
        rmse_group, bias_group, _, _ = self._analysis(ref, est, group_selection)
        nrmse_group, nbias_group, _, group_data = self._analysis(ref, est, group_selection, normalized=True)

        delta = ref - est
        norm = np.mean(ref)
        bias = np.mean(delta)
        rmse = np.sqrt(np.mean(delta ** 2))
        nbias = bias / norm
        nrmse = rmse / norm
        print("-" * 50, "Results Summery", "-" * 50)
        table = [["Metric", *[f"{gs}<r<{ge}" for gs, ge in group_selection]],
                 ["RMSE", *rmse_group],
                 ["BIAS", *bias_group],
                 ["NRMSE", *nrmse_group],
                 ["NBIAS", *nbias_group],
                 ["Total Metric", "Value", *["-" for _ in range(len(group_selection) - 1)]],
                 ["RMSE", rmse, *["-" for _ in range(len(group_selection) - 1)]],
                 ["BIAS", bias, *["-" for _ in range(len(group_selection) - 1)]],
                 ["NRMSE", nrmse, *["-" for _ in range(len(group_selection) - 1)]],
                 ["NBIAS", nbias, *["-" for _ in range(len(group_selection) - 1)]]
                 ]
        tab = PrettyTable(table[0])
        tab.add_rows(table[1:])
        print(tab)
        print("-" * 150)

        return nrmse_group, nbias_group, group_data
