import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_module("m", in_module)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.m(x) + self.scale * x


class MLP(nn.Sequential):
    def __init__(self, n_layer, input_size, output_size, intermediate_size, non_linear_function_class,
                 bias_output=True, normalization=None, residual=False):

        self.n_layer = n_layer
        self.input_size = input_size
        self.output_size = output_size
        self.intermediate_size = intermediate_size
        self.normalization = normalization
        self.non_linear_function_class = non_linear_function_class
        list_of_layers = []
        if self.n_layer == 1:
            list_of_layers.append(nn.Linear(self.input_size, self.output_size, bias=bias_output))
        else:
            for i in range(self.n_layer):
                if i == 0:
                    _list_of_layers = [nn.Linear(self.input_size, self.intermediate_size)]
                    if self.normalization is not None:
                        _list_of_layers.append(self.normalization(self.intermediate_size))
                    _list_of_layers.append(non_linear_function_class())
                    if residual:
                        list_of_layers.append(ResidualBlock(nn.Sequential(*_list_of_layers)))
                    else:
                        list_of_layers.append(nn.Sequential(*_list_of_layers))
                elif i == (self.n_layer - 1):
                    list_of_layers.append(nn.Linear(self.intermediate_size, self.output_size, bias=bias_output))
                else:
                    _list_of_layers = []
                    _list_of_layers.append(nn.Linear(self.intermediate_size, self.intermediate_size))
                    if self.normalization is not None:
                        _list_of_layers.append(self.normalization(self.intermediate_size))
                    _list_of_layers.append(non_linear_function_class())
                    if residual:
                        list_of_layers.append(ResidualBlock(nn.Sequential(*_list_of_layers)))
                    else:
                        list_of_layers.append(nn.Sequential(*_list_of_layers))
        super().__init__(*list_of_layers)
