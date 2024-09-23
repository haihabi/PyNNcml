from enum import Enum
from pynncml.neural_networks.nn_config import RNNType
import pkg_resources


class ModelType(Enum):
    """
    Model type

    ONESTEP: One step model
    TWOSTEP: Two steps model
    WETDRY: Wet dry model
    """
    ONESTEP = 0
    TWOSTEP = 1
    WETDRY = 2


MODEL_ZOO = {
    ModelType.ONESTEP: {RNNType.GRU: {1: 'one_step_1_rnntype_best.gru',
                                      2: 'one_step_2_rnntype_best.gru',
                                      3: 'one_step_3_rnntype_best.gru'},
                        RNNType.LSTM: {1: 'one_step_1_rnntype_best.lstm',
                                       2: 'one_step_2_rnntype_best.lstm',
                                       3: 'one_step_3_rnntype_best.lstm'}},
    ModelType.TWOSTEP: {RNNType.GRU: {1: 'two_step_1_rnntype_best.gru',
                                      2: 'two_step_2_rnntype_best.gru',
                                      3: 'two_step_3_rnntype_best.gru'},
                        RNNType.LSTM: {1: 'two_step_1_rnntype_best.lstm',
                                       2: 'two_step_2_rnntype_best.lstm',
                                       3: 'two_step_3_rnntype_best.lstm'}},
    ModelType.WETDRY: {RNNType.GRU: {1: 'wet_dry_1_rnntype_best.gru',
                                     2: 'wet_dry_2_rnntype_best.gru',
                                     3: 'wet_dry_3_rnntype_best.gru'},
                       RNNType.LSTM: {1: 'wet_dry_1_rnntype_best.lstm',
                                      2: 'wet_dry_2_rnntype_best.lstm',
                                      3: 'wet_dry_3_rnntype_best.lstm'}
                       }}


# TODO: Move to huggingface model zoo.
def get_model_from_zoo(model_type: ModelType, rnn_type: RNNType, n_layers: int) -> str:
    """
    Get the path to the model from the model zoo
    :param model_type: ModelType
    :param rnn_type: RNNType
    :param n_layers: int
    """
    if MODEL_ZOO.get(model_type) is None:
        raise Exception('unknown model:' + str(model_type))
    if MODEL_ZOO.get(model_type).get(rnn_type) is None:
        raise Exception('unknown RNN type:' + str(rnn_type))
    if MODEL_ZOO.get(model_type).get(rnn_type).get(n_layers) is None:
        raise Exception('there is not model with:' + str(n_layers) + 'layers')
    path2download = pkg_resources.resource_filename('pynncml',
                                                    'model_zoo/' + MODEL_ZOO.get(model_type).get(rnn_type).get(
                                                        n_layers))
    return path2download
