import torch

from pynncml.neural_networks import InputNormalizationConfig


def compute_data_normalization(in_data_loader, alpha: float = 0.9):
    """
    Compute the normalization parameters for the input data.
    :param in_data_loader: Data loader
    :param alpha: IIR filter alpha
    :return: InputNormalizationConfig
    """
    mean_dynamic = 0
    std_dynamic = 0
    mean_meta = 0
    std_meta = 0

    def iir_update(new, old):
        return (1 - alpha) * new + old * alpha

    for _, rsl, tsl, metadata in in_data_loader:
        _data = torch.cat([rsl, tsl], dim=-1)
        _data = _data.reshape([-1, 180])
        mean_dynamic = iir_update(_data.mean(dim=0), mean_dynamic)
        std_dynamic = iir_update(_data.std(dim=0), std_dynamic)
        mean_meta = iir_update(metadata.mean(dim=0), mean_meta)
        std_meta = iir_update(metadata.std(dim=0), std_meta)

    return InputNormalizationConfig(mean_dynamic.cpu().numpy(),
                                    std_dynamic.cpu().numpy(),
                                    mean_meta.cpu().numpy(),
                                    std_meta.cpu().numpy())
