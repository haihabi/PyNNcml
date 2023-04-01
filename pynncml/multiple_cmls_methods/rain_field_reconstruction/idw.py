import torch
from torch import nn
import numpy as np
from torch.nn import Parameter


class InverseDistanceWeighting(nn.Module):
    def __init__(self, in_h, in_w, modified=True, r=4, point_set=None, eps=1e-6):
        super(InverseDistanceWeighting, self).__init__()
        y_grid_vector = np.linspace(0, 1, in_h)
        x_grid_vector = np.linspace(0, 1, in_w)
        y_mesh, x_mesh = np.meshgrid(y_grid_vector, x_grid_vector)
        self.grid = Parameter(
            torch.tensor(
                np.expand_dims(np.expand_dims(np.stack([x_mesh, y_mesh], axis=0).astype('float32'), axis=0), axis=0)
            ),
            requires_grad=False)
        self.zero = Parameter(torch.tensor([0.0]).float(), requires_grad=False)
        self.p = 2
        self.modified = modified
        self.r = r
        self.eps = eps
        self.point_set = point_set
        if self.point_set is not None:
            self.point_set2weight()

    def point_set2weight(self):
        x = self.point_set.to_tensor()
        # rain_est in shape [N Sensors,1,N_Step] and link_set
        d = self._calculate_distance(x.unsqueeze(dim=0))
        # d = d * is_neg + 2 * (1 - is_neg)
        # D shape is [B,Sensors,32,32]
        is_zero = torch.isclose(d, self.zero)
        if self.modified:
            self.w = torch.pow(torch.relu(self.r - d) / (self.r * d), self.p)
        else:
            self.w = 1 / torch.pow(d, self.p)
        self.w[is_zero] = 1

    def forward(self, rain_est, link_set):
        if self.point_set is None:
            link_set.scale()
            self.point_set = link_set.center_point()
            self.point_set2weight()

        r_sensor = rain_est.T.reshape([rain_est.shape[1], rain_est.shape[0], 1, 1])
        # is_neg
        rain_map_non_zero = (r_sensor * self.w).sum(dim=1) / (self.w.sum(dim=1) + self.eps)
        # rain_map_zero = (is_neg * r_sensor * is_zero).sum(dim=1)
        return rain_map_non_zero  # add channel axis

    def _calculate_distance(self, x_i):
        x_i = x_i.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return torch.sqrt(torch.pow(x_i - self.grid, 2.0).sum(dim=2))
