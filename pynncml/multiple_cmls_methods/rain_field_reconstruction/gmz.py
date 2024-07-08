import math

import numpy as np
import torch

from pynncml.datasets import LinkSet
from pynncml.multiple_cmls_methods.rain_field_reconstruction.idw import InverseDistanceWeighting
from torch import nn

from pynncml.single_cml_methods.power_law.pl_module import a_b_parameters


def generate_link_set_gmz(in_link_set: LinkSet,
                          point_per_link: int = 3,
                          pixel_area: float = 1.0,
                          roi: float = 2,
                          modified=False,
                          r=4,
                          eps=1e-6):
    """
    Generate Inverse Distance Weighting weights for a set of links.
    :param in_link_set: LinkSet object
    :param point_per_link: Number of points per link
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
    point_list = []
    b_list = []
    for l in in_link_set:
        a, b = a_b_parameters(l.meta_data.frequency, l.meta_data.polarization)
        b_list.append(b)
        xy_array = l.meta_data.xy()
        x_points = np.linspace(xy_array[0], xy_array[2], point_per_link)
        y_points = np.linspace(xy_array[1], xy_array[3], point_per_link)
        x_norm_points = (x_points - in_link_set.x_min) / in_link_set.scale
        y_norm_points = (y_points - in_link_set.y_min) / in_link_set.scale

        point_list.append([x_norm_points, y_norm_points])
    point_list = np.asarray(point_list).transpose(0, 2, 1)

    base_idw = InverseDistanceWeighting(point_list.reshape([-1, 2]), x_grid_vector, y_grid_vector, roi=roi_eff,
                                        modified=modified,
                                        r=r, eps=eps)
    b_list = np.asarray(b_list)

    return GMZInterpolation(base_idw, b_list, point_per_link)


class GMZInterpolation(nn.Module):
    def __init__(self, base_idw: InverseDistanceWeighting, b_list, point_per_link: int = 3):
        """
        Inverse Distance Weighting module for rain field reconstruction.
        :param base_idw: Base IDW module
        :param b_list: B list
        :param point_per_link: Number of points per link
        """
        super(GMZInterpolation, self).__init__()
        self.base_idw = base_idw
        self.b_list = nn.Parameter(torch.tensor(b_list).float(), requires_grad=False)
        self.point_per_link = point_per_link

    def forward(self, rain_est):
        """
        Infer rain fields from a set of links.
        :param rain_est: Rain estimation tensor
        :return: Rain fields
        """
        _rain = rain_est.unsqueeze(dim=1).repeat(1, self.point_per_link, 1)  # Init rain point as copy of rain_est

        b_array = self.b_list.reshape([-1, 1, 1])
        _rain = _rain.reshape([-1, _rain.shape[-1]])
        loss_array = []
        rain_map = None
        for _ in range(10):
            rain_map = self.base_idw(_rain)
            rain_point_hat = self.compute_rain_point_from_field(rain_map)

            rain_point_hat = rain_point_hat.reshape([-1, self.point_per_link, rain_point_hat.shape[-1]])
            r_b = torch.pow(rain_point_hat, b_array)
            #################################
            # Just to compute loss
            #################################
            rain_link = _rain.reshape([-1, self.point_per_link, _rain.shape[-1]])

            loss = torch.sum((torch.pow(rain_link, b_array) - r_b) ** 2, dim=(0, 1))
            loss_array.append(loss.detach())
            #################################

            u = torch.pow(rain_est.unsqueeze(dim=1), b_array) - torch.mean(r_b, dim=1).unsqueeze(dim=1) + r_b
            u[u < 0] = 0
            _rain = torch.pow(u,
                              1 / self.b_list.reshape([-1, 1, 1]))
            _rain = _rain.reshape([-1, _rain.shape[-1]])
        if rain_map is None:
            raise ValueError("No rain map is computed")
        return rain_map, torch.stack(loss_array, dim=0)

    def compute_rain_point_from_field(self, in_rain_map):
        """
        Compute rain point from rain field.
        :param in_rain_map: Rain field
        :return: Rain point
        """
        point_set = self.base_idw.point_set
        point_set_x = point_set[:, 0]
        point_set_y = point_set[:, 1]
        x_grid_vector = self.base_idw.x_grid_vector
        y_grid_vector = self.base_idw.y_grid_vector

        delta_x = point_set_x.unsqueeze(dim=-1) - x_grid_vector.unsqueeze(dim=0)
        delta_y = point_set_y.unsqueeze(dim=-1) - y_grid_vector.unsqueeze(dim=0)
        i = torch.argmin(torch.abs(delta_x), dim=1)
        is_pos = (torch.gather(delta_x, dim=1, index=i.reshape(-1, 1)) > 0).long().flatten()
        i_floor = i * is_pos + (i - 1) * (1 - is_pos)
        i_ceiling = i * (1 - is_pos) + (i + 1) * is_pos

        j = torch.argmin(torch.abs(delta_y), dim=1)
        js_pos = (torch.gather(delta_y, dim=1, index=j.reshape(-1, 1)) > 0).long().flatten()
        j_floor = j * js_pos + (j - 1) * (1 - js_pos)
        j_ceiling = j * (1 - js_pos) + (j + 1) * js_pos

        ff = in_rain_map[:, i_floor, j_floor]
        cf = in_rain_map[:, i_ceiling, j_floor]
        fc = in_rain_map[:, i_floor, j_ceiling]
        cc = in_rain_map[:, j_ceiling, j_ceiling]

        w_x_floor = (point_set_x - x_grid_vector[i_floor]) / (x_grid_vector[i_ceiling] - x_grid_vector[i_floor])
        w_x_ceil = (x_grid_vector[i_ceiling] - point_set_x) / (x_grid_vector[i_ceiling] - x_grid_vector[i_floor])
        w_y_floor = (point_set_y - y_grid_vector[j_floor]) / (y_grid_vector[j_ceiling] - y_grid_vector[j_floor])
        w_y_ceil = (y_grid_vector[j_ceiling] - point_set_y) / (y_grid_vector[j_ceiling] - y_grid_vector[j_floor])

        rain_point = w_x_floor * w_y_floor * ff + w_x_ceil * w_y_floor * cf + w_x_floor * w_y_ceil * fc + w_x_ceil * w_y_ceil * cc

        return rain_point.T
