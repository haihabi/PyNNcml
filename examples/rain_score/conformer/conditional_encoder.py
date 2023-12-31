# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from pynncml.neural_networks.normalization import InputNormalization
from examples.rain_score.conformer.encoder import ConformerBlock
from .activation import Swish
from examples.rain_score.conformer.mlp import MLP


class OutputNormalization(nn.Module):
    def __init__(self, output_dim: int, beta=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mean", torch.zeros(output_dim))
        self.scale = nn.Parameter(torch.ones(1))
        self.count = 1
        self.beta = beta

    def forward(self, x):
        if self.training:
            mu = torch.mean(x, dim=0).detach()
            self.mean = self.mean * self.beta + (1 - self.beta) * mu
            self.count += 1

        _mean = self.mean / (1 - self.beta ** self.count)
        return self.scale * (x - self.mean.reshape([1, -1]))


class RainScoreBlock(nn.Module):
    def __init__(self, encoder_dim, theta_encoder_dim, exp_factor=1):
        super().__init__()
        self.feat_projection = nn.Linear(encoder_dim, 2 * theta_encoder_dim, bias=False)
        self.linear_one = nn.Linear(theta_encoder_dim, exp_factor * theta_encoder_dim)
        self.linear_two = nn.Linear(exp_factor * theta_encoder_dim, exp_factor * theta_encoder_dim)
        self.linear_three = nn.Linear(exp_factor * theta_encoder_dim, theta_encoder_dim, bias=False)
        self.normalization_one = nn.LayerNorm([theta_encoder_dim])
        self.normalization_two = nn.LayerNorm([theta_encoder_dim])
        self.scale_param = nn.Parameter(torch.ones(1) * 12, requires_grad=True)
        self.drop = nn.Dropout()
        self.bias = nn.Parameter(torch.ones(theta_encoder_dim) * 12, requires_grad=True)

        self.nl = Swish()

    def forward(self, input, cond):
        x = self.feat_projection(cond)
        x = x[:, -1, :]
        # print(x.shape)
        scale, shift = x.chunk(2, dim=-1)
        scale = torch.tanh(scale + self.bias.reshape([1, -1])) * self.scale_param

        input_cond_a = self.nl(self.normalization_one(self.linear_one(input)))
        input_cond_a = self.drop(input_cond_a)
        input_cond_a = self.linear_two(input_cond_a)
        input_cond_a = self.normalization_two(input_cond_a)
        input_cond_a = self.nl(input_cond_a)
        input_cond_a = self.drop(input_cond_a)

        res = input + scale * self.linear_three(input_cond_a) + shift

        return res


class RainScoreConformer(nn.Module):
    """
    Conformer encoder first processes the input with a convolution subsampling layer and then
    with a number of conformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
            self,
            input_dim: int = 180,
            meta_dim: int = 2,
            encoder_dim: int = 192,
            theta_encoder_dim: int = 128,
            num_layers: int = 2,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.0,
            attention_dropout_p: float = 0.0,
            conv_dropout_p: float = 0.0,
            conv_kernel_size: int = 7,
            half_step_residual: bool = True,
            normalization_cfg=None,
    ):
        super(RainScoreConformer, self).__init__()
        self.stem = nn.Conv1d(input_dim, encoder_dim, kernel_size=5, padding=2, padding_mode="zeros")
        self.lin_meta = nn.Linear(meta_dim, encoder_dim)

        self.mixing = nn.Linear(2 * encoder_dim, encoder_dim)

        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_layers)])

        self.cond_block = nn.ModuleList([RainScoreBlock(encoder_dim, theta_encoder_dim) for _ in range(num_layers)])
        self.cond_linear = nn.Linear(1, theta_encoder_dim)
        self.output = nn.Linear(theta_encoder_dim, 1, bias=False)
        self.normalization = InputNormalization(normalization_cfg)
        self.prior_net = MLP(3, 1, 1, 128, Swish, True, normalization=nn.LayerNorm)
        self.additive_step = MLP(3, 1, 1, 128, Swish, True, normalization=nn.LayerNorm)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_metadata: Tensor, input_cond_base: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_metadata (torch.FloatTensor): The length of input tensor. ``(batch)``
            input_cond_base (torch.FloatTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """

        inputs, input_metadata = self.normalization(inputs, input_metadata)
        inputs = torch.permute(inputs, [0, 2, 1])

        dyn = self.stem(inputs)
        meta = self.lin_meta(input_metadata)
        meta = torch.reshape(meta, [meta.shape[0], -1, 1]).repeat([1, 1, dyn.shape[-1]])
        fet = torch.cat([dyn, meta], dim=1)
        outputs = self.mixing(torch.permute(fet, [0, 2, 1]))
        input_cond_base = input_cond_base
        input_cond = self.cond_linear(input_cond_base)
        for layer, cond_block in zip(self.layers, self.cond_block):
            outputs = layer(outputs)
            input_cond = cond_block(input_cond, outputs)
        out_prior = self.prior_net(input_cond_base)
        out_add = self.additive_step(input_cond_base)
        return self.output(input_cond) + out_prior + out_add, out_prior
