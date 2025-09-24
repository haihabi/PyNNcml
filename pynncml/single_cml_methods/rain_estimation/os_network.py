import torch
import pynncml as pnc
from pynncml import neural_networks
from pynncml.datasets.alignment import  AttenuationType
from pynncml.cml_methods.base_cml_method import BaseCMLProcessingMethod
from pynncml.neural_networks.rnn_type_backbone import Backbone
from pynncml.neural_networks.rain_head import RainHead


class OneStepNetwork(BaseCMLProcessingMethod):
    """


    :param n_layers: integer that states the number of recurrent layers.
    :param rnn_type: enum that defines the type of the recurrent layer (GRU or LSTM).
    :param normalization_cfg: a class pnc.neural_networks.InputNormalizationConfig which hold the normalization parameters.
    :param enable_tn: boolean that enable or disable time normalization.
    :param tn_alpha: floating point number that defines the alpha factor of time normalization layer.
    :param rnn_input_size: int that represents the dynamic input size.
    :param rnn_n_features: int that represents the dynamic feature size.
    :param metadata_input_size: int that represents the metadata input size.
    :param metadata_n_features: int that represents the metadata feature size.
    :param input_data_type: enum that defines the type of the input data MinMax or Instance.
    :param input_rate: integer that represents the input rate in seconds.
    :param output_rate: integer that represents the output rate in seconds.
    """

    def __init__(self,
                 n_layers: int,
                 rnn_type: pnc.neural_networks.DNNType,
                 normalization_cfg: neural_networks.InputNormalizationConfig,
                 enable_tn: bool,
                 tn_alpha: float,
                 rnn_input_size: int,
                 rnn_n_features: int,
                 metadata_input_size: int,
                 metadata_n_features: int,
                 input_data_type:AttenuationType,
                 input_rate:int,
                 output_rate:int
                 ):
        super(OneStepNetwork, self).__init__(input_data_type,input_rate,output_rate)
        self.bb = Backbone(n_layers, rnn_type, normalization_cfg, enable_tn=enable_tn, tn_alpha=tn_alpha,
                           rnn_input_size=rnn_input_size, rnn_n_features=rnn_n_features,
                           metadata_input_size=metadata_input_size,
                           metadata_n_features=metadata_n_features)
        self.rh = RainHead(self.bb.total_n_features())

    def forward(self, data: torch.Tensor, metadata: torch.Tensor, state: torch.Tensor) -> (
            torch.Tensor, torch.Tensor):  # model forward pass
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
        return self.rh(features), state

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        This function generate the initial state of the Module.

        :param batch_size: int represent the batch size.
        :return: A Tensor, that hold the initial state.
        """
        return self.bb.init_state(batch_size=batch_size)
