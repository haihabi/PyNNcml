from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt

from pynncml.datasets.gauge_data import BasePointLocation
from pynncml.datasets.link_data import LinkBase


class PointSet:
    def __init__(self, gauge_set: List[BasePointLocation]):
        self.point_set = gauge_set

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor([[p.x, p.y] for p in self.point_set])

    @property
    def n_points(self):
        return len(self.point_set)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.n_points:
            p = self.point_set[self.n]
            self.n += 1
            return p
        else:
            raise StopIteration

    def plot_points(self):
        for p in self:
            plt.plot(p.x, p.y, "o", color="red")


class LinkSet:
    def __init__(self, link_list: List[LinkBase]):
        self.link_list = link_list
        self.scale_flag = False

    def area(self):
        xy_list = np.stack([l.meta_data.xy() for l in self])
        x = np.concatenate([xy_list[:, 0], xy_list[:, 2]])
        y = np.concatenate([xy_list[:, 1], xy_list[:, 3]])

        x_min = np.min(x)
        x_delta = np.max(x) - x_min

        y_min = np.min(y)
        y_delta = np.max(y) - y_min
        return np.sqrt(x_delta ** 2 + y_delta ** 2)

    def scale(self):
        if not self.scale_flag:
            self.scale_flag = True
            xy_list = np.stack([l.meta_data.xy() for l in self])
            x = np.concatenate([xy_list[:, 0], xy_list[:, 2]])
            y = np.concatenate([xy_list[:, 1], xy_list[:, 3]])
            x_min = np.min(x)
            x_delta = np.max(x) - x_min

            y_min = np.min(y)
            y_delta = np.max(y) - y_min
            for l in self:
                l.meta_data.update_scale(x_min, x_delta, y_min, y_delta)

    @property
    def n_links(self):
        return len(self.link_list)

    def center_point(self):
        return PointSet([BasePointLocation(*l.meta_data.xy_center()) for l in self])

    def get_link(self, link_index: int):
        if link_index > self.n_links or link_index < 0:
            raise Exception("illegal link index")
        return self.link_list[link_index]

    def plot_links(self):
        for link in self.link_list:
            xy_array = link.plot_link_position(self.scale_flag)
            plt.plot([xy_array[0], xy_array[2]], [xy_array[1], xy_array[3]], color="black")

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.n_links:
            link = self.link_list[self.n]
            self.n += 1
            return link
        else:
            raise StopIteration
