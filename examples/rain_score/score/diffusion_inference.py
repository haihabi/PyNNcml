import copy

import math
import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy.stats as stats
from torch import nn


class Domain(nn.Module):
    def __init__(self, lower, upper):
        super().__init__()
        self.register_buffer('upper', upper)
        self.register_buffer('lower', lower)

    def truncation(self, in_x):
        if self.lower == -torch.inf and self.upper == torch.inf:
            return in_x
        elif self.upper == torch.inf:
            return torch.maximum(in_x, self.lower)
        elif self.lower == -torch.inf:
            return torch.minimum(in_x, self.upper)
        else:
            return torch.maximum(torch.minimum(in_x, self.upper), self.lower)

    def reflection(self, in_x):
        return 2 * self.truncation(in_x) - in_x


def compute_gamma_factor(in_d, n_steps=1000, range_init=6, n_iterations=2):
    c = math.sqrt(2 * in_d)
    gamma_array = np.linspace(-range_init, range_init, n_steps)
    for _j in range(n_iterations):
        cost = norm.cdf(c * (gamma_array - 1) + 3 * gamma_array) - norm.cdf(c * (gamma_array - 1) - 3 * gamma_array)
        _i = np.argmin(np.abs(cost - 0.5))
        gamma_array = np.linspace(gamma_array[np.maximum(_i - 1, 0)], gamma_array[np.minimum(_i + 1, n_steps)], n_steps)

    return gamma_array[_i]


def generate_sigma_array(in_d, sigma_one, in_l):
    gamma = compute_gamma_factor(in_d)
    return [sigma_one * (gamma ** -_i) for _i in range(in_l)]


class DiffusionMLE:
    def __init__(self,
                 score_function,
                 d_p,
                 lr: float,
                 name: str,
                 n_iterations=1000,
                 beta: float = 0.1,
                 disable_noise=False,
                 in_domain: Domain = None,
                 adaptive_lr=False,
                 n_stages=6):
        self.adaptive_lr = adaptive_lr
        self.crb_true = None
        self.d_p = d_p
        self.score_function = score_function

        self.disable_noise = disable_noise
        self.domain = in_domain
        self.total_iteration = math.ceil(n_iterations)
        self.lr = lr
        self.name = name
        self.theta_true = None
        self.beta = torch.tensor(beta)
        self.gamma = torch.tensor(0.1)
        self.eps = 1e-4
        # n_stages = 5
        step_per_stages = math.ceil(n_iterations / n_stages)
        lr_scale_list = [10 ** (-i) for i in range(n_stages)]

        def lr_function(t):
            return lr_scale_list[math.floor(t / step_per_stages)]

        self.lr_function = lr_function

    @property
    def is_debug(self):
        return self.theta_true is not None

    def set_labels(self, **kwargs):
        self.theta_true = kwargs[self.name]

    def set_crb_matrix(self, in_crb_matrix):
        self.crb_true = in_crb_matrix

    def plot_debug(self, trails2plot=1, save_path=None):
        fig, axs = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [3, 1]})

        theta_array = np.stack(self.theta_list)[:, :, 0]
        mu = self.theta_true.detach().clone().cpu().numpy()[0, 0]
        axs[0].plot(theta_array[:, :trails2plot], color="red")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel(r"$\theta$")
        axs[0].grid()
        axs[0].set_xlim(0, theta_array.shape[0] - 1)
        # axs[0].set_ylim(-1, 1)
        # axs[0].fill_between(np.linspace(0, theta_array.shape[0] - 1, theta_array.shape[0]),
        #                     np.min(theta_array, axis=-1), np.max(theta_array, axis=-1))

        x = np.linspace(np.min(theta_array), np.max(theta_array), 1000)
        pdf_opt = stats.norm.pdf(x, mu, np.std(theta_array[-1, :]))
        pdf_act = stats.norm.pdf(x, np.mean(theta_array[-1, :]), np.std(theta_array[-1, :]))
        max_y = np.max([np.max(pdf_opt), np.max(pdf_act)])
        if self.crb_true is not None:
            axs[1].plot(stats.norm.pdf(x, mu, np.sqrt(self.crb_true[0, 0])), x, label="PDF (Target)")
            axs[1].plot([0, max_y], [mu, mu], label="Mean (Target)")
            axs[1].plot(stats.norm.pdf(x, np.mean(theta_array[-1, :]), np.std(theta_array[-1, :])), x,
                        label="PDF (Actual)")
            axs[1].plot([0, max_y], [np.mean(theta_array[-1, :]), np.mean(theta_array[-1, :])],
                        label="Mean (Actual)")

            axins = axs[1].inset_axes([0.3, 0.5, 0.67, 0.3])
            axins.set_ylim(mu - 3 * np.sqrt(self.crb_true[0, 0]), mu + 3 * np.sqrt(self.crb_true[0, 0]))
            axins.plot(stats.norm.pdf(x, mu, np.sqrt(self.crb_true[0, 0])), x, label="PDF (Target)")
            axins.plot([0, max_y], [mu, mu], label="Mean (Target)")
            axins.plot(stats.norm.pdf(x, np.mean(theta_array[-1, :]), np.std(theta_array[-1, :])), x,
                       label="PDF (Actual)")
            axins.plot([0, max_y], [np.mean(theta_array[-1, :]), np.mean(theta_array[-1, :])],
                       label="Mean (Actual)")
            axins.axes.yaxis.set_ticklabels([])
            axins.axes.xaxis.set_ticklabels([])

            axs[1].indicate_inset_zoom(axins, edgecolor="black")
            axs[1].grid()
            axs[1].legend()
            axs[1].set_xlabel("Probability")
            axs[1].axes.yaxis.set_ticklabels([])
            plt.tight_layout(pad=0.2)
        else:
            theta_true_array = self.theta_true.detach().clone().cpu().numpy().flatten()[:trails2plot]
            axs[1].plot(np.ones(theta_true_array.shape), theta_true_array, "o")
            # axs[1].axes.yaxis.set_ticklabels([])
            axs[1].grid()
            # axs[1].set_ylim(-1, 1)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def estimate(self, x, initialization_point=None, **kwargs):
        with torch.no_grad():
            theta_search = copy.deepcopy(kwargs)
            if initialization_point is None:
                theta_search[self.name] = torch.randn([x.shape[0], self.d_p]).to(x.device)
            else:
                theta_search[self.name] = torch.tensor(initialization_point).clone().to(x.device)
            if self.domain is not None:
                theta_search[self.name] = self.domain.reflection(theta_search[self.name])
            error_list = []
            alpha_list = []
            alpha_scale_list = []
            theta_list = [theta_search[self.name].detach().clone().cpu().numpy()]
            s_iir = torch.zeros([x.shape[0], self.d_p], device=x.device)
            for t in tqdm(range(self.total_iteration)):
                lr_scale = 1
                if self.lr_function is not None:
                    lr_scale = self.lr_function(t)

                s = self.score_function(x, **theta_search)  # Gradient function

                s_iir.mul_(self.gamma).add_(torch.abs(s).detach(), alpha=1 - self.gamma)
                bias_correction1 = 1 - torch.pow(self.gamma, t + 1)
                alpha_scale = bias_correction1 / (s_iir + self.eps)
                alpha_scale_list.append(alpha_scale.clone().detach().cpu().numpy())
                epsilon = torch.randn_like(s) * (1 - int(self.disable_noise))
                _lr = self.lr * lr_scale
                if self.adaptive_lr:
                    _lr *= alpha_scale

                theta_search[self.name].add_(_lr * s + torch.sqrt(2 * _lr / self.beta) * epsilon)
                if self.domain is not None:
                    theta_search[self.name] = self.domain.truncation(theta_search[self.name])
                theta_list.append(theta_search[self.name].detach().clone().cpu().numpy())

                if self.is_debug:
                    relative_error = torch.mean(
                        torch.norm(theta_search[self.name] - self.theta_true, dim=-1) / torch.norm(self.theta_true,
                                                                                                   dim=-1)).item()
                    error_list.append(relative_error)
            self.theta_list = theta_list
            if self.is_debug:
                e = (theta_search[self.name] - self.theta_true)
                return theta_search[self.name], error_list, e.T @ e / x.shape[0], alpha_list, alpha_scale_list

            return theta_search[self.name]
