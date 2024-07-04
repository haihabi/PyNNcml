import torch
import numpy as np
from torch import nn


class DynamicBaseLine(nn.Module):
    r"""
    The module impalement the dynamic baseline method that presented in [1] and defined as:
        .. math::
                A^{\Delta}_{i,n} =\min(A_{i,n},...,A_{i,n-k_s})

    where :math:`A^{\Delta}_{i,n}` is the module output and baseline value, :math:`A_{i,n}` is the input attenuation. The dynamic baseline method take the minimal value of :math:`k_s` consecutive samples.

    [1] J.Ostrometzky and H.Messer. "Dynamic Determination of the Baseline Level in Microwave Links for Rain Monitoring From Minimum Attenuation Values"

    :param k_steps: An integer, which state the number of step used for taking the minimal value.
    """

    def __init__(self, k_steps: int):
        super(DynamicBaseLine, self).__init__()
        self.k_steps = k_steps

    def forward(self, input_attenuation: torch.Tensor) -> torch.Tensor:  # model forward pass
        """
        The forward function of Dynamic Base line method

        :param input_attenuation: A Tensor of shape :math:`[N_b,N_s]` where :math:`N_b` is the batch size and :math:`N_s` is the length of time sequence. This parameter is the attenuation tensor symbolized as :math:`A_{i,n}`.
        :return: A Tensor of shape :math:`[N_b,N_s]` where :math:`N_b` is the batch size and :math:`N_s` is the length of time sequence. This parameter is the baseline tensor symbolized as :math:`A^{\Delta}_{i,n}`.
        """
        if len(input_attenuation.shape) != 2:
            raise Exception('Dynamic base line module only accepts inputs with shape length equal to 2')
        return torch.stack([self._single_link(input_attenuation[i, :]) for i in range(input_attenuation.shape[0])],
                           dim=0)

    def _single_link(self, input_attenuation: torch.Tensor) -> torch.Tensor:
        r"""
        The forward function of dynamic baseline of a single link.

        :param input_attenuation: A Tensor of shape :math:`[N_s]` where :math:`N_s` is the length of time sequence. This parameter is the attenuation tensor symbolized as :math:`A_{i,n}`.
        """
        return torch.stack([torch.min(input_attenuation[np.maximum(0, i - self.k_steps + 1): (i + 1)]) for i in
                            range(input_attenuation.shape[0])], dim=0)
