import torch
from torch import nn


def _single_link(attenuation: torch.Tensor, wd_classification: torch.Tensor):
    r"""
    The forward function of constant baseline.
    :param attenuation: A Tensor of shape :math:`[N_s]` where :math:`N_s` is the length of time sequence. This parameter is the attenuation tensor symbolized as :math:`A_{i,n}`.
    :param wd_classification: A Tensor of shape :math:`[N_s]` where :math:`N_s` is the length of time sequence. This parameter is the wet dry induction tensor symbolized as :math:`\hat{y}^{wd}_{i,n}`.

    """
    assert len(attenuation.shape) == 1
    baseline = [attenuation[0]]
    for i in range(1, attenuation.shape[0]):
        if wd_classification[i]:
            baseline.append(baseline[i - 1])
        else:
            baseline.append(attenuation[i])
    return torch.stack(baseline, dim=0)


class ConstantBaseLine(nn.Module):
    r"""
            This is the module is implantation of Constant baseline that presented in [1] and defined as:

            .. math::
                    A^{\Delta}_{i,n} =
                    \begin{cases}
                        A_{i,n},& \text{if } \hat{y}^{wd}_{i,n} = 0\\
                        A^{\Delta}_{i,n-1},              & \text{otherwise}
                    \end{cases}

            where :math:`A^{\Delta}_{i,n}` is the module output and baseline value, :math:`A_{i,n}` is the input attenuation and  :math:`\hat{y}^{wd}_{i,n}` is the predicition of wet-dry classification used as indicator.

            [1] Schleiss, Marc and Berne, Alexis. "Identification of dry and rainy periods using telecommunication microwave links"

    """

    def __init__(self):
        super(ConstantBaseLine, self).__init__()

    def forward(self, input_attenuation: torch.Tensor, input_wet_dry: torch.Tensor) -> torch.Tensor:
        r"""
        The forward function of constant baseline.

        :param input_attenuation: A Tensor of shape :math:`[N_b,N_s]` where :math:`N_b` is the batch size and :math:`N_s` is the length of time sequence. This parameter is the attenuation tensor symbolized as :math:`A_{i,n}`.
        :param input_wet_dry: A Tensor of shape :math:`[N_b,N_s]` where :math:`N_b` is the batch size and :math:`N_s` is the length of time sequence. This parameter is the wet dry induction tensor symbolized as :math:`\hat{y}^{wd}_{i,n}`.
        :return: A Tensor of shape :math:`[N_b,N_s]` where :math:`N_b` is the batch size and :math:`N_s` is the length of time sequence. This parameter is the baseline tensor symbolized as :math:`A^{\Delta}_{i,n}`.
        """
        return torch.stack(
            [_single_link(input_attenuation[batch_index, :], input_wet_dry[batch_index, :]) for batch_index in
             range(input_attenuation.shape[0])], dim=0)
