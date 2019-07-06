import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchrain.neural_networks import InputNormalizationConfig


class InputNormalization(nn.Module):
    r"""

    """
    def __init__(self, config: InputNormalizationConfig):
        super(InputNormalization, self).__init__()
        self.mean_dynamic = Parameter(torch.as_tensor(config.mean_dynamic).float(), requires_grad=False)
        self.std_dynamic = Parameter(torch.as_tensor(config.std_dynamic).float(), requires_grad=False)
        self.mean_metadata = Parameter(torch.as_tensor(config.mean_metadata).float(), requires_grad=False)
        self.std_metadata = Parameter(torch.as_tensor(config.std_metadata).float(), requires_grad=False)

    def forward(self, x, metadata):
        x_norm = (x - self.mean_dynamic) / self.std_dynamic
        metadata_norm = (metadata - self.mean_metadata) / self.std_metadata
        return x_norm, metadata_norm
