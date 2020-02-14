import torch
import pynncml as pnc
from pynncml.rain_estimation.ts_constant import TwoStepConstant
from pynncml.rain_estimation.os_dynamic import OneStepDynamic
from pynncml.rain_estimation.os_network import OneStepNetwork
from pynncml.rain_estimation.ts_network import TwoStepNetwork
from pynncml.model_common import get_model_from_zoo, ModelType


def two_step_constant_baseline(power_law_type: pnc.power_law.PowerLawType, r_min: float, window_size: int,
                               threshold: float, wa_factor: float = None):
    if wa_factor is None:
        return TwoStepConstant(power_law_type, r_min, window_size, threshold)
    else:
        return TwoStepConstant(power_law_type, r_min, window_size, threshold, wa_factor=wa_factor)


def one_step_dynamic_baseline(power_law_type: pnc.power_law.PowerLawType, r_min: float, window_size: int):
    return OneStepDynamic(power_law_type, r_min, window_size)


def two_step_network(n_layers: int, rnn_type: pnc.neural_networks.RNNType,
                     normalization_cfg: pnc.neural_networks.InputNormalizationConfig = pnc.neural_networks.INPUT_NORMALIZATION,
                     enable_tn: bool = False,
                     tn_alpha: float = 0.9,
                     tn_affine: bool = False,
                     rnn_input_size: int = pnc.neural_networks.DYNAMIC_INPUT_SIZE,
                     rnn_n_features: int = pnc.neural_networks.RNN_FEATURES,
                     metadata_input_size: int = pnc.neural_networks.STATIC_INPUT_SIZE,
                     metadata_n_features: int = pnc.neural_networks.FC_FEATURES, pretrained=True):
    """


    :param n_layers: integer that state the number of recurrent layers.
    :param rnn_type: enum that define the type of the recurrent layer (GRU or LSTM).
    :param normalization_cfg: a class tr.neural_networks.InputNormalizationConfig which hold the normalization parameters.
    :param enable_tn: boolean that enable or disable time normalization.
    :param tn_alpha: floating point number which define the alpha factor of time normalization layer.
    :param tn_affine: boolean that state if time normalization have affine transformation.
    :param rnn_input_size: int that represent the dynamic input size.
    :param rnn_n_features: int that represent the dynamic feature size.
    :param metadata_input_size: int that represent the metadata input size.
    :param metadata_n_features: int that represent the metadata feature size.
    :param pretrained: boolean flag state that state if to download a pretrained model.
    """
    model = TwoStepNetwork(n_layers, rnn_type, normalization_cfg, enable_tn=enable_tn, tn_alpha=tn_alpha,
                           tn_affine=tn_affine,
                           rnn_input_size=rnn_input_size, rnn_n_features=rnn_n_features,
                           metadata_input_size=metadata_input_size, metadata_n_features=metadata_n_features)
    if pretrained and not enable_tn:
        model_file = get_model_from_zoo(ModelType.TWOSTEP, rnn_type, n_layers)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    return model


def one_step_network(n_layers: int, rnn_type: pnc.neural_networks.RNNType,
                     normalization_cfg: pnc.neural_networks.InputNormalizationConfig = pnc.neural_networks.INPUT_NORMALIZATION,
                     enable_tn: bool = False,
                     tn_alpha: float = 0.9,
                     tn_affine: bool = False,
                     rnn_input_size: int = pnc.neural_networks.DYNAMIC_INPUT_SIZE,
                     rnn_n_features: int = pnc.neural_networks.RNN_FEATURES,
                     metadata_input_size: int = pnc.neural_networks.STATIC_INPUT_SIZE,
                     metadata_n_features: int = pnc.neural_networks.FC_FEATURES,
                     pretrained=True
                     ):
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
    model = OneStepNetwork(n_layers, rnn_type, normalization_cfg, enable_tn=enable_tn, tn_alpha=tn_alpha,
                           tn_affine=tn_affine,
                           rnn_input_size=rnn_input_size, rnn_n_features=rnn_n_features,
                           metadata_input_size=metadata_input_size, metadata_n_features=metadata_n_features)
    if pretrained and not enable_tn:
        model_file = get_model_from_zoo(ModelType.ONESTEP, rnn_type, n_layers)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    return model
