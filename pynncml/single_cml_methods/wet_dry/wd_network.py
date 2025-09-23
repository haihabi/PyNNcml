import torch
import pynncml as pnc
from torch import nn
from pynncml.neural_networks.backbone import Backbone
from pynncml.neural_networks.wd_head import WetDryHead
from pynncml import neural_networks


class WetDryNetwork(nn.Module):
    """
    The module generate a wet dry network that was presented in [1]

    :param n_layers: integer that state the number of recurrent layers.
    :param rnn_type: enum that define the type of the recurrent layer (GRU or LSTM).
    :param normalization_cfg: a class pnc.neural_networks.InputNormalizationConfig which
                              hold the normalization parameters.
    :param enable_tn: boolean that enable or disable time normalization.
    :param tn_alpha: floating point number which define the alpha factor of time normalization layer.
    :param tn_affine: boolean that state if time normalization have affine transformation.
    :param rnn_input_size: int that represent the dynamic input size.
    :param rnn_n_features: int that represent the dynamic feature size.
    :param metadata_input_size: int that represent the metadata input size.
    :param metadata_n_features: int that represent the metadata feature size.
    """

    def __init__(self, n_layers: int, rnn_type: pnc.neural_networks.DNNType,
                 normalization_cfg: neural_networks.InputNormalizationConfig,
                 enable_tn: bool,
                 tn_alpha: float,
                 tn_affine: bool,
                 rnn_input_size: int,
                 rnn_n_features: int,
                 metadata_input_size: int,
                 metadata_n_features: int,
                 ):
        super(WetDryNetwork, self).__init__()
        self.bb = Backbone(n_layers, rnn_type, normalization_cfg, enable_tn=enable_tn, tn_alpha=tn_alpha,
                           rnn_input_size=rnn_input_size, rnn_n_features=rnn_n_features,
                           metadata_input_size=metadata_input_size,
                           metadata_n_features=metadata_n_features)
        self.wdh = WetDryHead(self.bb.total_n_features())

    def forward(self, data: torch.Tensor, metadata: torch.Tensor, state: torch.Tensor) -> (
            torch.Tensor, torch.Tensor):
        """
        This is the module forward function

        :param data: A tensor of the dynamic data of shape :math:`[N_b,N_s,N_i^d]` where :math:`N_b` is the batch size,
                     :math:`N_s` is the length of time sequence and :math:`N_i^d` is the dynamic input size.
        :param metadata:  A tensor of the metadata of shape :math:`[N_b,N_i^m]` where :math:`N_b` is the batch size,
                          and :math:`N_i^m` is the metadata input size.
        :param state: A tensor that represent the state of shape
        :return: Two Tensors, the first tensor if the feature tensor of size :math:`[N_b,N_s,N_f]`
                    where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence
                    and :math:`N_f` is the number of feature.
                    The second tensor is the state tensor.
        """

        features, state = self.bb(data, metadata, state)
        return self.wdh(features), state

    def init_state(self, batch_size: int = 1) -> torch.Tensor:  # model init state\
        """
        This function generate the initial state of the Module.

        :param batch_size: int represent the batch size.
        :return: A Tensor, that hold the initial state.
        """
        return self.bb.init_state(batch_size=batch_size)
