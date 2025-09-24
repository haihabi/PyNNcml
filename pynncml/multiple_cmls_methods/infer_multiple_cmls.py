import torch
from torch import nn

from pynncml.datasets import LinkSet


class InferMultipleCMLs(nn.Module):
    """
    Infer rain fields from a set of links.

    """

    def __init__(self, in_cml2rain_method: callable,is_recurrent=False,is_attenuation=True):
        """
        Infer rain fields from a set of links.
        :param in_cml2rain_method: Method to infer rain from a link

        """
        super().__init__()
        self.cml2rain = in_cml2rain_method
        self.is_recurrent=is_recurrent
        self.is_attenuation=is_attenuation

    def forward(self, link_set: LinkSet) -> (torch.Tensor, torch.Tensor):
        """
        Infer rain fields from a set of links.
        :param link_set: Set of links
        :return: Rain fields
        """
        res_list = []
        for link in link_set:
            if self.is_attenuation:
                data = link.attenuation()
            else:
                data = link.as_tensor()
            if self.is_recurrent:
                state=self.cml2rain.init_state(1)
                rain_est = self.cml2rain(data, link.meta_data.as_tensor(),state)  # .T.unsqueeze(dim=0)
            else:
                rain_est = self.cml2rain(data, link.meta_data)  # .T.unsqueeze(dim=0)
            if isinstance(rain_est, tuple):
                rain_est = rain_est[0]
            res_list.append(rain_est.flatten())
        link_results = torch.stack(res_list, dim=0)
        return link_results
