import torch
from pynncml.neural_networks.rain_gan import DCGANGenerator
from huggingface_hub import hf_hub_url, cached_download
from pynncml.utils import get_working_device


def get_rain_filed_generation_function(h, w):
    working_device = get_working_device()
    net_g = DCGANGenerator(128, h, w, z_size=128, condition_vector_size=2).to(
        working_device)
    hf_url = hf_hub_url("HVH/RainGAN", "RainGAN32x32.pt")
    file_path = cached_download(hf_url)
    net_g.load_state_dict(torch.load(file_path, map_location='cpu'))
    net_g.eval()

    def sample_rain_field(rain_coverage, n_peaks, peak_rain_rate, batch_size=1):
        with torch.no_grad():
            cond = torch.Tensor([rain_coverage, n_peaks]).reshape([1, -1]).repeat([batch_size, 1])
            z = torch.randn([batch_size, 128])
            return peak_rain_rate * net_g(z, cond).cpu().numpy()[:, 0, :, :]

    return sample_rain_field
