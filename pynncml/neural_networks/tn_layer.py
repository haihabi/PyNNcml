import torch
from torch import nn
from torch.nn.parameter import Parameter


class TimeNormalization(nn.Module):
    r"""
    Time Normalization Layer, normalized the input using it's mean and variance over time as defined in the follow equation:
        .. math::
            \hat{h}_n=\frac{h_n-\mu_n}{\sqrt{\sigma_n^2+\epsilon}}\\
            \mu_n=\alpha h_n +(1-\alpha)\mu_{n-1}


    :param alpha: A float, which represent the alpha parameter in the equation.
    :param num_features: An integer, which represent the number of features in the input tensor.
    :param epsilon: A float, which represent the epsilon parameter in the equation.
    :param affine: A boolean, which represent if the layer should learn an affine transformation after normalization.
    """

    def __init__(self, alpha: float, num_features: int, epsilon: float = 0.001):
        super(TimeNormalization, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.alpha = alpha
        self.one_minus_alpha = 1 - self.alpha

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> (
            torch.Tensor, torch.Tensor):
        """
        This is the module forward function.

        :param x: A tensor of the input data of shape :math:`[N_b,N_s,N_f]` where :math:`N_b` is the batch size,
                    :math:`N_s` is the length of time sequence and :math:`N_f` is the number of features.
        :param state: A tensor that represent the state of shape :math:`[2,N_b,N_f]` where :math:`N_b` is the batch size and :math:`N_f` is the number of features.

        :return: Two Tensors, the first tensor if the normalized tensor of size :math:`[N_b,N_s,N_f]`
        """
        p = self.alpha * torch.stack([x, torch.pow(x, 2)], dim=0)
        ##########################################################
        # Loop over all time steps
        ##########################################################
        res_list_state = []
        for i in range(x.shape[1]):  # This may slow the
            state = p[:, :, i, :] + self.one_minus_alpha * state
            res_list_state.append(state)

        state_vector = torch.stack(res_list_state, dim=2)  # stack all time steps
        mean_vector = state_vector[0, :, :, :]
        var_vector = state_vector[1, :, :, :] - torch.pow(mean_vector, 2)  # build variance vector
        x_norm = (x - mean_vector) / torch.sqrt(var_vector + self.epsilon)  # normlized x vector
        return x_norm, state

    def init_state(self, working_device, batch_size: int = 1) -> torch.Tensor:
        """
        This function generate the initial state of the Module. This include only Time Normalization state

        :param working_device: str that state the current working device.
        :param batch_size: int represent the batch size.
        :return: A Tensor, that hold the initial state.
        """
        state = torch.stack(
            [torch.zeros(batch_size, self.num_features, device=working_device),
             torch.ones(batch_size, self.num_features, device=working_device)],
            dim=0)
        return state
