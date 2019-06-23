import torch
import torch.nn as nn
from torchrain import nn_common


# from torch_helpers import change2torch_tensor
# from models.rain_rate_models.robusness_layer import robustness_normalization


class Backbone(nn.Module):
    def __init__(self, n_layers: nn_common.NLayers, rnn_type: nn_common.RNNType):
        super(Backbone, self).__init__()
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        # Model Layers
        if rnn_type == nn_common.RNNType.GRU:
            self.rnn = nn.GRU(nn_common.DYNAMIC_INPUT_SIZE, nn_common.RNN_FEATURES,
                              bidirectional=False, num_layers=1 + n_layers.value,
                              batch_first=True)
        elif rnn_type == nn_common.RNNType.LSTM:
            self.rnn = nn.LSTM(nn_common.DYNAMIC_INPUT_SIZE, nn_common.RNN_FEATURES,
                               bidirectional=False, num_layers=1 + n_layers.value,
                               batch_first=True)
        else:
            raise Exception('Unknown RNN type')

        self.fc_meta = nn.Linear(nn_common.STATIC_INPUT_SIZE, nn_common.FC_FEATURES)
        self.relu = nn.ReLU()

    def forward(self, input_tensor, input_meta_tensor, hidden):  # model forward pass

        output_meta = self.relu(self.fc_meta(input_meta_tensor))
        output, hidden = self.rnn(input_tensor, hidden)
        output = output.contiguous()

        output_size = output.size()
        output_meta = output_meta.view(output_meta.size()[0], 1, output_meta.size()[1]).repeat(1, output.size()[1], 1)
        ##############################################################################
        # if self.config.get('re'):  # split hidden state for RE
        #     output_new, mean_var_state = robustness_normalization(output, mean_var_state, self.config)
        # else:
        #     output_new = output
        ##############################################################################
        features = torch.cat([output, output_meta], dim=-1)

        return features, hidden

    def init_state(self, batch_size=1):  # model init state
        return torch.zeros(int(self.n_layers.value) + 2 * self.config.get('re'),
                           batch_size, nn_common.RNN_FEATURES, device=self.config.device)
