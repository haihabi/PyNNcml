from pynncml.cml_methods.apis.xarray_processing.xarray_inference_engine import XarrayInferenceEngine
from pynncml.neural_networks import DNNType
from pynncml.single_cml_methods.wet_dry import wet_dry_network,statistics_wet_dry


def create_wet_dry_dnn_gru(n_layers=2,
                       rnn_type=DNNType.GRU):
    nn_base=wet_dry_network(n_layers=n_layers,rnn_type=rnn_type)
    return XarrayInferenceEngine(nn_base)

def create_wet_dry_std(threshold=2.3,step=10):
    nn_base=statistics_wet_dry(threshold,step)
    return XarrayInferenceEngine(nn_base,is_recurrent=False,is_attenuation=True)


if __name__ == '__main__':
    nn_xarray=create_wet_dry_dnn_gru()