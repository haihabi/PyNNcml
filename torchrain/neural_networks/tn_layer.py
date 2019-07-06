import torch
from torch import nn
from torch.nn.parameter import Parameter


class TimeNormalization(nn.Module):
    def __init__(self, alpha: float, num_features: int, epsilon: float = 0.001, affine: bool = False):
        """
        Time Normalization Layer

        $$ $$

        :param alpha:
        :param num_features:
        :param epsilon:
        :param affine:
        """
        super(TimeNormalization, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.alpha = alpha
        self.one_minus_alpha = 1 - self.alpha
        self.affine = affine
        self.weight = Parameter(torch.Tensor(1, 1, num_features))
        self.bias = Parameter(torch.Tensor(1, 1, num_features))

    def forward(self, x: torch.Tensor, state: torch.Tensor, indicator: torch.Tensor = None) -> (
    torch.Tensor, torch.Tensor):
        p = self.alpha * torch.stack([x, torch.pow(x, 2)], dim=0)
        ##########################################################
        # Loop over all time steps
        ##########################################################
        res_list_state = []
        if indicator is None:
            for i in range(x.shape[1]):  # This may slow the
                state = p[:, :, i, :] + self.one_minus_alpha * state
                res_list_state.append(state)
        else:
            for i in range(x.shape[1]):  # This may slow the
                ind = indicator[:, i].repeat(2, 1).unsqueeze(dim=-1)
                state = ind * (p[:, :, i, :] + self.one_minus_alpha * state) + (1 - ind) * state
                res_list_state.append(state)
        state_vector = torch.stack(res_list_state, dim=2)  # stack all time steps
        mean_vector = state_vector[0, :, :, :]
        var_vector = state_vector[1, :, :, :] - torch.pow(mean_vector, 2)  # build variance vector
        x_norm = (x - mean_vector) / torch.sqrt(var_vector + self.epsilon)  # normlized x vector
        if self.affine:  # apply affine transformation
            x_norm = self.weight * x_norm + self.bias
        return x_norm, state

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        This function generate the initial state of the Module. This include only Time Normalization state

        :param batch_size: int represent the batch size.
        :return: A Tensor, that hold the initial state.
        """
        state = torch.stack(
            [torch.zeros(batch_size, self.num_features, device=self.weight.device.type),
             torch.ones(batch_size, self.num_features, device=self.weight.device.type)],
            dim=0)
        return state
