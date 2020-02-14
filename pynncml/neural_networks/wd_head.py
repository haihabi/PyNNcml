import torch
import torch.nn as nn


class WetDryHead(nn.Module):
    r"""
    The Wet Dry head module, perform a linear operation on the input feature vector followed by a sigmoid function.

    :param n_features: the input feature vector size.
    """

    def __init__(self, n_features: int):
        super(WetDryHead, self).__init__()
        self.fc = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        This is the module forward function.

        :param input_tensor: A tensor of the features of shape :math:`[N_b,N_s,N_f]` where :math:`N_b` is the batch size,
                     :math:`N_s` is the length of time sequence and :math:`N_f` is the number of features.
        :return: A Tensor of size :math:`[N_b,N_s,1]`
                    where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence
                    and probability of rain.
        """
        return self.sigmoid(self.fc(input_tensor))
