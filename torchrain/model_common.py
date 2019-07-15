import os
import urllib.request
from enum import Enum
from torchrain.neural_networks.nn_config import RNNType


class ModelType(Enum):
    ONESTEP = 0
    TWOSTEP = 1
    WETDRY = 2


DOWNLOAD_URL = {
    ModelType.ONESTEP: {RNNType.GRU: {1: 'https://drive.google.com/open?id=13CYBpV6OFqM94Ae__KNyZ3HBxE_aokFK',
                                      2: 'https://drive.google.com/open?id=1fS7lT1X9lPR57_kprzQwh3TisShJXAB2',
                                      3: 'https://drive.google.com/open?id=1N5CYOwY3CeQBU0hlYSboKLoZSP8SwChT'},
                        RNNType.LSTM: {1: 'https://drive.google.com/open?id=1I-Bayv2rbKjgMROcOKmPg7oTdmsNIhj4',
                                       2: 'https://drive.google.com/open?id=1Lae0IQkKiDArDEkYbHM_K1G6TFQXDckI',
                                       3: 'https://drive.google.com/open?id=12kqCdSn9ZgR07YC9iJhaum1bK5CYSEkI'}},
    ModelType.TWOSTEP: {RNNType.GRU: {1: 'https://drive.google.com/open?id=1doQtVC5_CrhYdfPrz48y6V0oSIk1U7x_',
                                      2: 'https://drive.google.com/open?id=1qWJJpJ9SEYFD6w7JRgc2hkQFFVLzBx6B',
                                      3: 'https://drive.google.com/open?id=1KjvkngOwMfuUDy1a99ACm7inZffnwG_O'},
                        RNNType.LSTM: {1: 'https://drive.google.com/open?id=1owDak5IKALa3wD0maw9pe2U-f9VsSNTp',
                                       2: 'https://drive.google.com/open?id=1Usy1Yhec_0Ky_lP5TO731esLtPCM9jUJ',
                                       3: 'https://drive.google.com/open?id=1biUHVdOeWOvCfEu9U87NB8O3Ctdifh-Q'}},
    ModelType.WETDRY: {RNNType.GRU: {1: 'https://drive.google.com/open?id=1UmLQe9zem4TRyB84qtKpgCr4jYi0nCdB',
                                     2: 'https://drive.google.com/open?id=1DFRkx9kt9MSy6epI41b7c_SBiytKrMQs',
                                     3: 'https://drive.google.com/open?id=1kBXr_nF26D4l86QrL5BhD6xSkOHDqcNP'},
                       RNNType.LSTM: {1: 'https://drive.google.com/open?id=1OUvn9EuV_zd5meWTVgO-aAUHtZaNKqR-',
                                      2: 'https://drive.google.com/open?id=1ZXeA_8819AgyAFaOViYAUeo-nW1Z9xiG',
                                      3: 'https://drive.google.com/open?id=1IpXUVxrqWATTuYfFIk3XzHwwc0ziJ79l'}
                       }}


def download_model(model_type: ModelType, rnn_type: RNNType, n_layers: int, download_path: str = './') -> str:
    if DOWNLOAD_URL.get(model_type) is None:
        raise Exception('unknown model:' + str(model_type))
    if DOWNLOAD_URL.get(model_type).get(rnn_type) is None:
        raise Exception('unknown RNN type:' + str(rnn_type))
    if DOWNLOAD_URL.get(model_type).get(rnn_type).get(n_layers) is None:
        raise Exception('there is not model with:' + str(n_layers) + 'layers')
    path2download = os.path.join(download_path, model_type.name + '_' + rnn_type.name + '_' + str(n_layers) + '.model')
    urllib.request.urlretrieve(DOWNLOAD_URL.get(model_type).get(rnn_type).get(n_layers), path2download)
    return path2download
