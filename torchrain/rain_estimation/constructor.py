import torchrain as tr
from torchrain.rain_estimation.ts_constant import TwoStepConstant
from torchrain.rain_estimation.os_dynamic import OneStepDynamic
from torchrain.rain_estimation.os_network import OneStepNetwork
from torchrain.rain_estimation.ts_network import TwoStepNetwork


def two_step_constant_baseline(power_law_type: tr.power_law.PowerLawType, r_min: float, window_size: int,
                               threshold: float, wa_factor: float = None):
    if wa_factor is None:
        return TwoStepConstant(power_law_type, r_min, window_size, threshold)
    else:
        return TwoStepConstant(power_law_type, r_min, window_size, threshold, wa_factor=wa_factor)


def one_step_dynamic_baseline(power_law_type: tr.power_law.PowerLawType, r_min: float, window_size: int):
    return OneStepDynamic(power_law_type, r_min, window_size)


def two_step_network(n_layers: int, rnn_type: tr.neural_networks.RNNType,
                     enable_tn: bool = False,
                     tn_alpha: float = 0.9,
                     tn_affine: bool = False,
                     rnn_input_size: int = tr.neural_networks.DYNAMIC_INPUT_SIZE,
                     rnn_n_features: int = tr.neural_networks.RNN_FEATURES,
                     metadata_input_size: int = tr.neural_networks.STATIC_INPUT_SIZE,
                     metadata_n_features: int = tr.neural_networks.FC_FEATURES):
    return TwoStepNetwork(n_layers, rnn_type, enable_tn=enable_tn, tn_alpha=tn_alpha, tn_affine=tn_affine,
                          rnn_input_size=rnn_input_size, rnn_n_features=rnn_n_features,
                          metadata_input_size=metadata_input_size, metadata_n_features=metadata_n_features)


def one_step_network(n_layers: int, rnn_type: tr.neural_networks.RNNType,
                     enable_tn: bool = False,
                     tn_alpha: float = 0.9,
                     tn_affine: bool = False,
                     rnn_input_size: int = tr.neural_networks.DYNAMIC_INPUT_SIZE,
                     rnn_n_features: int = tr.neural_networks.RNN_FEATURES,
                     metadata_input_size: int = tr.neural_networks.STATIC_INPUT_SIZE,
                     metadata_n_features: int = tr.neural_networks.FC_FEATURES):
    return OneStepNetwork(n_layers, rnn_type, enable_tn=enable_tn, tn_alpha=tn_alpha, tn_affine=tn_affine,
                          rnn_input_size=rnn_input_size, rnn_n_features=rnn_n_features,
                          metadata_input_size=metadata_input_size, metadata_n_features=metadata_n_features)
