import torch
from torch import nn


class DCGANGenerator(nn.Module):
    """
    Generator for DCGAN from https://github.com/haihabi/RainMapGenerator/blob/main/networks/dcgan.py.

    """

    def __init__(self, dim, h, w, out_features=16, z_size=128, condition_vector_size=0):
        """
        :param dim: Dimension of the model
        :param h: Height of the input image
        :param w: Width of the input image
        :param out_features: Number of output features
        :param z_size: Size of the input noise vector
        :param condition_vector_size: Size of the conditional vector
        """
        super(DCGANGenerator, self).__init__()
        self.z_size = z_size
        self.h_in = int(h / 8)
        self.w_in = int(w / 8)
        self.h_mid = int(h / 4)
        self.w_mid = int(w / 4)
        self.h = h
        self.w = w
        self.condition_vector_size = condition_vector_size
        self.conditional = self.condition_vector_size > 0

        self.out_features = out_features
        norm_class = nn.Identity
        preprocess = nn.Sequential(
            nn.Linear(self.z_size + self.condition_vector_size, 4 * self.h_in * self.w_in * dim),
            nn.ELU(),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, (4, 4), stride=2, padding=1),
            norm_class(),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, (4, 4), stride=2, padding=1),
            norm_class(),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(dim, self.out_features, (4, 4), stride=2, padding=1)

        self.output_intensity = nn.Conv2d(self.out_features, 1, kernel_size=1, stride=1, padding=0)
        self.dim = dim
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.eps = 1e-6
        self.output_nl = nn.Sigmoid()

    def forward(self, input_tensor, cond=None):
        """
        A forward pass of the generator.

        :param input_tensor: Input tensor
        :param cond: Conditional tensor
        :return: Output tensor
        """
        if self.conditional:
            input_tensor = torch.cat([input_tensor, cond], dim=-1)
        output = self.preprocess(input_tensor)
        output = output.view(-1, 4 * self.dim, self.h_in, self.w_in)
        output = self.block1(output)  # x2 8,8
        output = self.block2(output)  # x2 16,16
        output = self.deconv_out(output)
        output_intensity = self.output_nl(self.output_intensity(output))
        output = output_intensity
        return output.view(-1, 1, self.h, self.w)
