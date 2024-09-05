import torch
from torch import nn

from pynncml.datasets import LinkSet


class InferMultipleCMLs(nn.Module):
    """
    Infer rain fields from a set of links.

    """

    def __init__(self, in_cml2rain_method: callable):
        """
        Infer rain fields from a set of links.
        :param in_cml2rain_method: Method to infer rain from a link

        """
        super().__init__()
        self.cml2rain = in_cml2rain_method

    def forward(self, link_set: LinkSet) -> (torch.Tensor, torch.Tensor):
        """
        Infer rain fields from a set of links.
        :param link_set: Set of links
        :return: Rain fields
        """
        res_list = []
        for link in link_set:
            rain_est = self.cml2rain(link.attenuation().T.unsqueeze(dim=0), link.meta_data)
            res_list.append(rain_est.flatten())
        link_results = torch.stack(res_list, dim=0)
        return link_results
