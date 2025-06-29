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
        """
        Get the next point in the set.
        :return: PointSensor
        """
        if self.n < self.n_points:
            p = self.point_set[self.n]
            self.n += 1
            return p
        else:
            raise StopIteration

    def plot_points(self):
        """
        Plot the points in the set.
        """
        for p in self:
            plt.plot(p.x, p.y, "o", color="red")

    def find_near_gauge(self, xy_center):
        """
        Find the nearest gauge to the center point.
        :param xy_center: Center point
        """
        d_list = [math.sqrt((xy_center[0] - g.x) ** 2 + (xy_center[1] - g.y) ** 2) for g in self.point_set]
        return np.min(d_list), [self.point_set[np.argmin(d_list)]]

    def find_near_gauges(self, xy_center, range_limit):
        """
        Find the nearest gauges to the center point within a given range.
        :param xy_center: Center point
        :param range_limit: Range limit for searching gauges
        :return: List of gauges within the range
        """
        d_list = [math.sqrt((xy_center[0] - g.x) ** 2 + (xy_center[1] - g.y) ** 2) for g in self.point_set]
        gauge_with_distance = [(self.point_set[i], d) for i, d in enumerate(d_list) if d < range_limit]
        sorted_gauge_with_distance = sorted(gauge_with_distance, key=lambda x: x[1])
        return [g[1] for g in sorted_gauge_with_distance], [g[0] for g in sorted_gauge_with_distance]


class LinkSet:
    def __init__(self, link_list: List[LinkBase]):
        """
        Data structure that contains a set of links.
        :param link_list: List of links

        """
        self.link_list = link_list
        self.max_label_size = max([l.number_of_labels() for l in self.link_list])
        if len(self.link_list) == 0:
            raise Exception("Link set is empty")
        pair_list = self.find_link_pairs()

        self.pair_list = pair_list
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
        self.scale = np.maximum(x_delta, y_delta)

    def find_link_pairs(self):
        found_list_pair = []
        pair_list = []
        # Find pairs of links that are the same but in different directions
        for link in self.link_list:
            if link not in found_list_pair:
                xy = link.meta_data.xy()
                for _link in self.link_list:
                    if (_link.meta_data.xy()[0] == xy[2] and
                            _link.meta_data.xy()[2] == xy[0] and
                            _link.meta_data.xy()[1] == xy[3] and
                            _link.meta_data.xy()[3] == xy[1] and
                            _link != link):
                        found_list_pair.append(_link)
                        found_list_pair.append(link)
                        pair_list.append((link, _link))
        return pair_list

    def __len__(self):
        """
        Get the number of links
        """
        return self.n_links

    def area(self):
        """
        Get the area of the link set
        :return: float
        """
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
        """
        Number of links in the set.
        """
        return len(self.link_list)

    def center_point(self, scale=False):
        """
        Get the center point of the link set.
        :param scale: Scale the center point
        :return: Center points
        """
        point_list = []
        for l in self:
            x, y = l.meta_data.xy_center()
            if scale:
                x = (x - self.x_min) / self.scale
                y = (y - self.y_min) / self.scale
            point_list.append([x, y])
        return point_list

    def get_link(self, link_index: int):
        """
        Get the link at the given index.
        :param link_index: Index of the link
        """
        if link_index > self.n_links or link_index < 0:
            raise Exception("illegal link index")
        return self.link_list[link_index]

    def plot_links(self, scale: bool = False, scale_factor: float = 1.0, scale_y: float = 1.0):
        """
        Plot the links in the set.
        :param scale: Scale the link
        :param scale_factor: Scale factor
        """
        index = 0
        gauge2index = {}
        for link in self.link_list:
            xy_array = link.plot_link_position()
            if scale:
                xy_array[0] = scale_factor * (xy_array[0] - self.x_min) / self.scale
                xy_array[2] = scale_factor * (xy_array[2] - self.x_min) / self.scale
                xy_array[1] = scale_y * (xy_array[1] - self.y_min) / self.scale
                xy_array[3] = scale_y * (xy_array[3] - self.y_min) / self.scale

            if link.gauge_ref is None:
                plt.plot([xy_array[0], xy_array[2]], [xy_array[1], xy_array[3]], color="black")
            else:
                gauge_near = None
                if isinstance(link.gauge_ref, PointSensor):
                    gauge_near = link.gauge_ref
                elif isinstance(link.gauge_ref, list):
                    gauge_near = link.gauge_ref[0]
                if gauge2index.get(gauge_near) is None:
                    gauge2index.update({gauge_near: index})
                    index = index + 1
                plt.plot([xy_array[0], xy_array[2]], [xy_array[1], xy_array[3]],
                         color=COLOR_LIST[gauge2index[gauge_near]])
        for g, i in gauge2index.items():
            if self.scale:
                plt.plot(scale_factor * (g.x - self.x_min) / self.scale, scale_y * (g.y - self.y_min) / self.scale,
                         "o", color=COLOR_LIST[i])
            else:
                plt.plot(g.x, g.y, "o", color=COLOR_LIST[i])

    def __iter__(self):
        """
        Initialize the iterator.
        """
        self.n = 0
        return self

    def __next__(self):
        """
        Get the next link in the set.
        :return: LinkBase
        """
        if self.n < self.n_links:
            link = self.link_list[self.n]
            self.n += 1
            return link
        else:
            raise StopIteration
