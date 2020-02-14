import torch
from torch import nn
from torch.nn.parameter import Parameter
from pynncml.neural_networks import InputNormalizationConfig


class InputNormalization(nn.Module):
    r"""
    This module normalized both dynamic and static(metadata) data, using the following two equations:
        .. math::
            \bar{x}_d=\frac{x_d-\mu_d}{\sigma_d}\\
            \bar{x}_{s}=\frac{x_s-\mu_s}{\sigma_s}

    :param config: the input normalization config which hold the mean and the standard deviation for both dynamic and static(metadata) data.

    """

    def __init__(self, config: InputNormalizationConfig):
        super(InputNormalization, self).__init__()
        self.mean_dynamic = Parameter(torch.as_tensor(config.mean_dynamic).float(), requires_grad=False)
        self.std_dynamic = Parameter(torch.as_tensor(config.std_dynamic).float(), requires_grad=False)
        self.mean_metadata = Parameter(torch.as_tensor(config.mean_metadata).float(), requires_grad=False)
        self.std_metadata = Parameter(torch.as_tensor(config.std_metadata).float(), requires_grad=False)

    def forward(self, data: torch.Tensor, metadata: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        This is the module forward function.

        :param data: A tensor of the dynamic data of shape :math:`[N_b,N_s,N_i^d]` where :math:`N_b` is the batch size,
                     :math:`N_s` is the length of time sequence and :math:`N_i^d` is the dynamic input size.
        :param metadata:  A tensor of the metadata of shape :math:`[N_b,N_i^m]` where :math:`N_b` is the batch size,
                          and :math:`N_i^m` is the metadata input size.
        :return: Two Tensors, the first tensor if the feature tensor of size :math:`[N_b,N_s,N_f]`
                    where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence
                    and :math:`N_f` is the number of feature.
                    The second tensor is the state tensor.
        """
        x_norm = (data - self.mean_dynamic) / self.std_dynamic
        metadata_norm = (metadata - self.mean_metadata) / self.std_metadata
        return x_norm, metadata_norm
