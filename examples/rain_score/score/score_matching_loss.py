import torch
import torch.autograd as autograd
from torch.nn.modules.loss import _Loss


def batch_vec(in_x):
    return torch.reshape(in_x, [in_x.shape[0], -1])


def make_require_grad(*args):
    return [x * torch.ones(x.shape, requires_grad=True, device=x.device) for x in args]


def score_norm(in_score, scale=None):
    if scale is None:
        return 0.5 * (torch.norm(in_score, dim=-1)) ** 2
    else:
        return 0.5 * torch.sum((in_score ** 2) * scale, dim=-1)


def batch_dot_product(in_x, in_y):
    return torch.sum(in_x * in_y, dim=-1)


def batch_compute_jac(in_score, in_data, scale=None):
    grad_list = []
    for i in range(in_score.shape[1]):
        if scale is None:
            outputs = in_score[:, i]
        else:
            outputs = in_score[:, i] * scale[:, i]
        gradients = autograd.grad(outputs=outputs, inputs=in_data,
                                  grad_outputs=torch.ones(in_score[:, i].size(), requires_grad=True).to(
                                      in_score.device),
                                  create_graph=True, retain_graph=True)[0]
        grad_list.append(gradients)
    return torch.stack(grad_list, dim=-1)


class ScoreMatchingLoss(_Loss):
    def __init__(self, min_value, max_value, reg=True):
        super().__init__()
        if torch.is_tensor(max_value):
            max_value = torch.tensor(max_value)

        if torch.is_tensor(min_value):
            min_value = torch.tensor(min_value)
        self.register_buffer("min_value", min_value)

        self.register_buffer("max_value", max_value)
        self.reg = reg

    def forward(self, in_score, in_data):
        scale = torch.minimum(in_data - self.min_value, self.max_value - in_data)
        jac = batch_compute_jac(in_score, in_data, scale=scale)
        # reg_ms = 0
        # if self.reg:
        #     _jac = batch_compute_jac(in_score, in_data, scale=None)
        #     j = in_score.shape[0]
        #     matrix_b = torch.mean(
        #         in_score.reshape([in_score.shape[0], -1, 1]) @ in_score.reshape([in_score.shape[0], 1, -1]), dim=0)
        #     matrix_a = torch.mean(_jac, dim=0)
        #     g = -matrix_b @ torch.linalg.inv(matrix_a)
        #     p = g.shape[0]
        #     d = -1 + torch.trace(g) / p
        #     t = -1 + torch.pow(torch.linalg.det(g), 1 / p)
        #     reg_ms = j * (p ** 2) * ((t ** 2) + (d ** 2)) / 2
        #     loss_metric = torch.diag(matrix_a) / torch.sqrt(torch.diag(matrix_b))
        jac_trace = torch.diagonal(jac, dim1=1, dim2=2).sum(dim=-1)
        return torch.mean(score_norm(in_score, scale=scale) + jac_trace)
