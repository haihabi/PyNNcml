import math
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt

from pynncml.datasets.gauge_data import PointSensor
from pynncml.datasets.link_data import LinkBase

COLOR_LIST = ["blue",
              "green",
              "red",
              "cyan",
              "purple",
              "pink",
              "brown",
              "gray",
              "olive",
              "orange"]


class PointSet:
    def __init__(self, gauge_set: List[PointSensor]):
        """
        Data structure that contains a set of points.
        :param gauge_set: List of points
        """
        self.point_set = gauge_set

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the point set to a tensor.
        :return: Tensor of shape [n_points, 2]
        """
        return torch.Tensor([[p.x, p.y] for p in self.point_set])

    @property
    def n_points(self):
        """
        Number of points in the set.
        """
        return len(self.point_set)

    def __iter__(self):
        """
        Initialize the iterator.
        """
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

    def find_near_gauge(self, xy_center):
        d_list = [math.sqrt((xy_center[0] - g.x) ** 2 + (xy_center[1] - g.y) ** 2) for g in self.point_set]
        return np.min(d_list), self.point_set[np.argmin(d_list)]


class LinkSet:
    def __init__(self, link_list: List[LinkBase]):
        """
        Data structure that contains a set of links.
        :param link_list: List of links

        """
        self.link_list = link_list

        xy_list = np.stack([l.meta_data.xy() for l in self])
        x = np.concatenate([xy_list[:, 0], xy_list[:, 2]])
        y = np.concatenate([xy_list[:, 1], xy_list[:, 3]])
        x_min = np.min(x)
        x_delta = np.max(x) - x_min

        y_min = np.min(y)
        y_delta = np.max(y) - y_min

        self.x_min = x_min
        self.x_delta = x_delta
        self.y_min = y_min
        self.y_delta = y_delta
        # self.scale = np.sqrt(x_delta ** 2 + y_delta ** 2)
        self.scale = np.maximum(x_delta, y_delta)

    def __len__(self):
        return self.n_links

    def area(self):
        xy_list = np.stack([l.meta_data.xy() for l in self])
        x = np.concatenate([xy_list[:, 0], xy_list[:, 2]])
        y = np.concatenate([xy_list[:, 1], xy_list[:, 3]])

        x_min = np.min(x)
        x_delta = np.max(x) - x_min

        y_min = np.min(y)
        y_delta = np.max(y) - y_min
        return np.sqrt(x_delta ** 2 + y_delta ** 2)

    @property
    def n_links(self):
        return len(self.link_list)

    def center_point(self, scale=False):
        point_list = []
        for l in self:
            x, y = l.meta_data.xy_center()
            if scale:
                x = (x - self.x_min) / self.scale
                y = (y - self.y_min) / self.scale
            point_list.append([x, y])
        return point_list

    def get_link(self, link_index: int):
        if link_index > self.n_links or link_index < 0:
            raise Exception("illegal link index")
        return self.link_list[link_index]

    def plot_links(self, scale=False, scale_factor=1.0):
        index = 0
        gauge2index = {}
        for link in self.link_list:
            xy_array = link.plot_link_position()
            if scale:
                xy_array[0] = scale_factor * (xy_array[0] - self.x_min) / self.scale
                xy_array[2] = scale_factor * (xy_array[2] - self.x_min) / self.scale
                xy_array[1] = scale_factor * (xy_array[1] - self.y_min) / self.scale
                xy_array[3] = scale_factor * (xy_array[3] - self.y_min) / self.scale

            if link.gauge_ref is None:
                plt.plot([xy_array[0], xy_array[2]], [xy_array[1], xy_array[3]], color="black")
            else:
                if gauge2index.get(link.gauge_ref) is None:
                    gauge2index.update({link.gauge_ref: index})
                    index = index + 1
                plt.plot([xy_array[0], xy_array[2]], [xy_array[1], xy_array[3]],
                         color=COLOR_LIST[gauge2index[link.gauge_ref]])
        for g, i in gauge2index.items():
            if self.scale:
                plt.plot(scale_factor*(g.x - self.x_min) / self.scale,scale_factor* (g.y - self.y_min) / self.scale, "o", color=COLOR_LIST[i])
            else:
                plt.plot(g.x, g.y, "o", color=COLOR_LIST[i])

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
