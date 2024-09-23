import torch
from torch import nn
import numpy as np
from torch.nn import Parameter
import math

from pynncml.datasets import LinkSet


def generate_link_set_idw(in_link_set: LinkSet, pixel_area: float = 1.0, roi: float = 2, modified=False, r=4, eps=1e-6):
    """
    Generate Inverse Distance Weighting weights for a set of links.
    :param in_link_set: LinkSet object
    :param pixel_area: Pixel area in km^2
    :param roi: Radius of influence in km
    :param modified: Use modified IDW or not
    :param r: R value for modified IDW
    :param eps: Epsilon value to avoid division by zero
    :return: Weights tensor
    """
    y_n_grid = math.ceil(in_link_set.y_delta / (pixel_area * 1000))
    x_n_grid = math.ceil(in_link_set.x_delta / (pixel_area * 1000))
    if in_link_set.y_delta < in_link_set.x_delta:
        y_grid_vector = np.linspace(0, in_link_set.y_delta / in_link_set.x_delta, y_n_grid)
        x_grid_vector = np.linspace(0, 1, x_n_grid)
    else:
        y_grid_vector = np.linspace(0, 1, y_n_grid)
        x_grid_vector = np.linspace(0, in_link_set.x_delta / in_link_set.y_delta, x_n_grid)
    roi_eff = roi / (in_link_set.scale / 1000)
    measurement_locations = in_link_set.center_point(scale=True)
    return InverseDistanceWeighting(measurement_locations, x_grid_vector, y_grid_vector, roi=roi_eff, modified=modified,
                                    r=r, eps=eps)


class InverseDistanceWeighting(nn.Module):
    def __init__(self, measurement_locations, x_grid_vector, y_grid_vector, roi: float = 2, modified=False, r=4,
                 eps=1e-6):
        """
        Inverse Distance Weighting module for rain field reconstruction.
        :param measurement_locations: Measurement locations tensor.
        :param x_grid_vector: X grid vector.
        :param y_grid_vector: Y grid vector.
        :param roi: Radius of influence.
        :param modified: Use modified IDW or not
        :param r: R value for modified IDW
        :param eps: Epsilon value to avoid division by zero
        """
        super(InverseDistanceWeighting, self).__init__()

        self.y_grid_vector = nn.Parameter(torch.tensor(y_grid_vector).float(), requires_grad=False)
        self.x_grid_vector = nn.Parameter(torch.tensor(x_grid_vector).float(), requires_grad=False)
        y_mesh, x_mesh = np.meshgrid(self.y_grid_vector, self.x_grid_vector)
        self.grid = Parameter(
            torch.tensor(
                np.expand_dims(np.expand_dims(np.stack([x_mesh, y_mesh], axis=0).astype('float32'), axis=0), axis=0)
            ),
            requires_grad=False)
        self.zero = Parameter(torch.tensor([0.0]).float(), requires_grad=False)

        self.p = 2
        self.modified = modified
        self.r = r
        self.roi = roi
        self.eps = eps
        self.point_set = nn.Parameter(torch.tensor(measurement_locations).float(), requires_grad=False)
        self.w = self.point_set2weight()

    def point_set2weight(self):
        """
        Calculate the weights for the points in the point set

        """
        x = self.point_set
        d = self._calculate_distance(x.unsqueeze(dim=0))
        if self.modified:
            w = torch.pow(torch.relu(self.r - d) / (self.r * d), self.p)
        else:
            w = 1 / (torch.pow(d, self.p) + self.eps)
        w[d > self.roi] = 0
        return w

    def forward(self, rain_est):
        """
        Infer rain fields from a set of links.
        :param rain_est: Rain estimation tensor
        :return: Rain fields
        """

        r_sensor = rain_est.T.reshape([rain_est.shape[1], rain_est.shape[0], 1, 1])
        rain_map_non_zero = (r_sensor * self.w).sum(dim=1) / (self.w.sum(dim=1) + self.eps)
        return rain_map_non_zero  # add channel axis

    def _calculate_distance(self, x_i):
        """
        Calculate the distance between the points in the point set and the grid points
        :param x_i: Point set tensor
        """
        x_i = x_i.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return torch.sqrt(torch.pow(x_i - self.grid, 2.0).sum(dim=2))
