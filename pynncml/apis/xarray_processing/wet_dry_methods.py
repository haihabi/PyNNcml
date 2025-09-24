from pynncml.apis.xarray_processing.xarray_inference_engine import XarrayInferenceEngine
from pynncml.neural_networks import DNNType
from pynncml.single_cml_methods.wet_dry import wet_dry_network


def create_wet_dry_dnn(n_layers=2,
                       rnn_type=DNNType.GRU):
    nn_base=wet_dry_network(n_layers=n_layers,rnn_type=rnn_type)
    return XarrayInferenceEngine(nn_base)

if __name__ == '__main__':
    nn_xarray=create_wet_dry_dnn()