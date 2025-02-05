import torch
from pynncml.neural_networks.rain_gan import DCGANGenerator
from huggingface_hub import hf_hub_url, hf_hub_download
from pynncml.utils import get_working_device


def get_rain_filed_generation_function(h: int, w: int, working_device=None) -> callable:
    """
    Get a function that generates a rain field using the RainGAN model.
    :param h: Height of the rain field
    :param w: Width of the rain field
    :param working_device: Device to use for generating the rain field
    :return: Function that generates a rain field
    """
    if working_device is None:
        working_device = get_working_device()
    net_g = DCGANGenerator(128, h=32, w=32, z_size=128, condition_vector_size=2).to(
        working_device)
    net_g.eval()

    class RainGAN(torch.nn.Module):
        def __init__(self, in_net_g, h, w):
            super(RainGAN, self).__init__()
            self.net_g = in_net_g
            self.resize = torch.nn.Upsample(size=(h, w), mode='bilinear', align_corners=False)

        def forward(self, z, cond):
            return self.resize(self.net_g(z, cond))

    rain_gan = RainGAN(net_g, h, w)

    file_path = hf_hub_download("HVH/RainGAN", "RainGAN32x32.pt")
    net_g.load_state_dict(torch.load(file_path, map_location='cpu'))
    net_g.eval()

    def sample_rain_field(rain_coverage: float, n_peaks: int, peak_rain_rate: float, batch_size: int = 1):
        """
        Generate a rain field.
        :param rain_coverage: Rain coverage
        :param n_peaks: Number of peaks
        :param peak_rain_rate: Peak rain rate
        :param batch_size: Batch size
        :return: Rain field
        """
        with torch.no_grad():
            cond = torch.tensor([rain_coverage, n_peaks], device=working_device).reshape([1, -1]).repeat(
                [batch_size, 1])
            z = torch.randn([batch_size, 128], device=working_device)
            return peak_rain_rate * rain_gan(z, cond).cpu().numpy()[:, 0, :, :]

    return sample_rain_field
