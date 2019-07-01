import torch
import torchrain as tr
from torch import nn
from torchrain.neural_networks.backbone import Backbone
from torchrain.neural_networks.rain_head import RainHead
from torchrain.neural_networks.wd_head import WetDryHead
from torchrain import neural_networks


class TwoStepNetwork(nn.Module):
    def __init__(self, n_layers: int, rnn_type: tr.neural_networks.RNNType,
                 normalization_cfg: neural_networks.InputNormalizationConfig,
                 enable_tn: bool,
                 tn_alpha: float,
                 tn_affine: bool,
                 rnn_input_size: int,
                 rnn_n_features: int,
                 metadata_input_size: int,
                 metadata_n_features: int,
                 ):
        super(TwoStepNetwork, self).__init__()
        self.bb = Backbone(n_layers, rnn_type, normalization_cfg, enable_tn=enable_tn, tn_alpha=tn_alpha,
                           rnn_input_size=rnn_input_size,
                           tn_affine=tn_affine, rnn_n_features=rnn_n_features, metadata_input_size=metadata_input_size,
                           metadata_n_features=metadata_n_features)
        self.rh = RainHead(self.bb.total_n_features())
        self.wdh = WetDryHead(self.bb.total_n_features())

    def forward(self, data: torch.Tensor, metadata: tr.MetaData,
                state: torch.Tensor) -> (torch.Tensor, torch.Tensor):  # model forward pass
        features, state = self.bb(data, metadata.as_tensor(), state)
        return torch.cat([self.rh(features), self.wdh(features)], dim=-1), state

    def init_state(self, batch_size: int = 1) -> torch.Tensor:  # model init state
        return self.bb.init_state(batch_size=batch_size)
