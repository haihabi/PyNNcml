import torch
import pynncml as pnc
from pynncml.single_cml_methods.wet_dry.std_wd import STDWetDry
from pynncml.single_cml_methods.wet_dry.wd_network import WetDryNetwork
from pynncml.model_zoo.model_common import get_model_from_zoo, ModelType


def statistics_wet_dry(th, step, is_min_max):
    return STDWetDry(th, step, is_min_max)


def wet_dry_network(n_layers: int, rnn_type: pnc.neural_networks.RNNType,
                    normalization_cfg: pnc.neural_networks.InputNormalizationConfig = pnc.neural_networks.INPUT_NORMALIZATION,
                    enable_tn: bool = False,
                    tn_alpha: float = 0.9,
                    tn_affine: bool = False,
                    rnn_input_size: int = pnc.neural_networks.DYNAMIC_INPUT_SIZE,
                    rnn_n_features: int = pnc.neural_networks.RNN_FEATURES,
                    metadata_input_size: int = pnc.neural_networks.STATIC_INPUT_SIZE,
                    metadata_n_features: int = pnc.neural_networks.FC_FEATURES,
                    pretrained=True):
    """


    :param n_layers: integer that state the number of recurrent layers.
    :param rnn_type: enum that define the type of the recurrent layer (GRU or LSTM).
    :param normalization_cfg: a class pnc.neural_networks.InputNormalizationConfig which hold the normalization parameters.
    :param enable_tn: boolean that enable or disable time normalization.
    :param tn_alpha: floating point number which define the alpha factor of time normalization layer.
    :param tn_affine: boolean that state if time normalization have affine transformation.
    :param rnn_input_size: int that represent the dynamic input size.
    :param rnn_n_features: int that represent the dynamic feature size.
    :param metadata_input_size: int that represent the metadata input size.
    :param metadata_n_features: int that represent the metadata feature size.
    :param pretrained: boolean flag state that state if to download a pretrained model.
    """
    model = WetDryNetwork(n_layers, rnn_type, normalization_cfg, enable_tn=enable_tn, tn_alpha=tn_alpha,
                          tn_affine=tn_affine,
                          rnn_input_size=rnn_input_size, rnn_n_features=rnn_n_features,
                          metadata_input_size=metadata_input_size, metadata_n_features=metadata_n_features)
    if pretrained and not enable_tn:
        model_file = get_model_from_zoo(ModelType.WETDRY, rnn_type, n_layers)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    return model
