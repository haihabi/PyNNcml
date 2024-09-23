import torch

from pynncml.neural_networks import InputNormalizationConfig, RNNType, INPUT_NORMALIZATION, STATIC_INPUT_SIZE, \
    DYNAMIC_INPUT_SIZE, RNN_FEATURES, FC_FEATURES
from pynncml.single_cml_methods.power_law import PowerLawType
from pynncml.single_cml_methods.rain_estimation.ts_constant import TwoStepsConstant
from pynncml.single_cml_methods.rain_estimation.os_dynamic import OneStepDynamic
from pynncml.single_cml_methods.rain_estimation.os_network import OneStepNetwork
from pynncml.single_cml_methods.rain_estimation.ts_network import TwoStepNetwork
from pynncml.model_zoo.model_common import get_model_from_zoo, ModelType


def two_step_constant_baseline(power_law_type: PowerLawType, r_min: float, window_size: int,
                               threshold: float, wa_factor: float = None):
    """
    This function create a two step constant baseline model. The model is used to estimate the rain rate from the CML data.


    :param power_law_type: enum that define the type of the power law.
    :param r_min: floating point number that represent the minimum value of the rain rate.
    :param window_size: integer that represent the window size.
    :param threshold: floating point number that represent the threshold value.
    :param wa_factor: floating point number that represent the weight average factor.
    """
    if wa_factor is None:
        return TwoStepsConstant(power_law_type, r_min, window_size, threshold)
    else:
        return TwoStepsConstant(power_law_type, r_min, window_size, threshold, wa_factor=wa_factor)


def one_step_dynamic_baseline(power_law_type: PowerLawType, r_min: float, window_size: int, quantization_delta: float):
    """
    This function create a one step dynamic baseline model. The model is used to estimate the rain rate from the CML data.
    This function also includes the quantization delta parameter for bias correction.

    :param power_law_type: enum that define the type of the power law.
    :param r_min: floating point number that represent the minimum value of the rain rate.
    :param window_size: integer that represent the window size.
    :param quantization_delta: floating point number that represent the quantization delta.
    """
    return OneStepDynamic(power_law_type, r_min, window_size, quantization_delta)


def two_step_network(n_layers: int, rnn_type: RNNType,
                     normalization_cfg: InputNormalizationConfig = INPUT_NORMALIZATION,
                     enable_tn: bool = False,
                     tn_alpha: float = 0.9,
                     tn_affine: bool = False,
                     rnn_input_size: int = DYNAMIC_INPUT_SIZE,
                     rnn_n_features: int = RNN_FEATURES,
                     metadata_input_size: int = STATIC_INPUT_SIZE,
                     metadata_n_features: int = FC_FEATURES,
                     pretrained=True):
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


def one_step_network(n_layers: int,
                     rnn_type: RNNType,
                     normalization_cfg: InputNormalizationConfig = INPUT_NORMALIZATION,
                     enable_tn: bool = False,
                     tn_alpha: float = 0.9,
                     tn_affine: bool = False,
                     rnn_input_size: int = DYNAMIC_INPUT_SIZE,
                     rnn_n_features: int = RNN_FEATURES,
                     metadata_input_size: int = STATIC_INPUT_SIZE,
                     metadata_n_features: int = FC_FEATURES,
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
    model = OneStepNetwork(n_layers, rnn_type, normalization_cfg, enable_tn=enable_tn, tn_alpha=tn_alpha,
                           tn_affine=tn_affine,
                           rnn_input_size=rnn_input_size, rnn_n_features=rnn_n_features,
                           metadata_input_size=metadata_input_size, metadata_n_features=metadata_n_features)
    if pretrained and not enable_tn:
        model_file = get_model_from_zoo(ModelType.ONESTEP, rnn_type, n_layers)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    return model
