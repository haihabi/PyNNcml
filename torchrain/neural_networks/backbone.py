import torch
import torch.nn as nn
from torchrain import neural_networks


class Backbone(nn.Module):
    def __init__(self, n_layers: int, rnn_type: neural_networks.RNNType,
                 enable_tn: bool,
                 tn_alpha: float,
                 tn_affine: bool,
                 rnn_input_size: int,
                 rnn_n_features: int,
                 metadata_input_size: int,
                 metadata_n_features: int,
                 ):
        super(Backbone, self).__init__()
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.metadata_n_features = metadata_n_features
        self.rnn_n_features = rnn_n_features
        # Model Layers
        if rnn_type == neural_networks.RNNType.GRU:
            self.rnn = nn.GRU(rnn_input_size, rnn_n_features,
                              bidirectional=False, num_layers=n_layers,
                              batch_first=True)
        elif rnn_type == neural_networks.RNNType.LSTM:
            self.rnn = nn.LSTM(rnn_input_size, rnn_n_features,
                               bidirectional=False, num_layers=n_layers,
                               batch_first=True)
        else:
            raise Exception('Unknown RNN type')
        self.enable_tn = enable_tn
        if enable_tn:
            self.tn = neural_networks.TimeNormalization(alpha=tn_alpha, num_features=rnn_n_features,
                                                        affine=tn_affine)
        self.fc_meta = nn.Linear(metadata_input_size, metadata_n_features)
        self.relu = nn.ReLU()

    def total_n_features(self) -> int:
        return self.metadata_n_features + self.rnn_n_features

    def forward(self, input_tensor, input_meta_tensor, hidden):  # model forward pass
        if self.enable_tn:  # split hidden state for RE
            hidden_tn = hidden[1]
            hidden_rnn = hidden[0]
        else:
            hidden_rnn = hidden
            hidden_tn = None
        output_meta = self.relu(self.fc_meta(input_meta_tensor))
        output, hidden_rnn = self.rnn(input_tensor, hidden_rnn)
        output = output.contiguous()
        output_meta = output_meta.view(output_meta.size()[0], 1, output_meta.size()[1]).repeat(1, output.size()[1], 1)
        ##############################################################################
        if self.enable_tn:  # run TimeNormalization over rnn output and update the state of Backbone
            output_new, hidden_tn = self.tn(output, hidden_tn)
            hidden = (hidden_rnn, hidden_tn)
        else:  # pass rnn state and output
            output_new = output
            hidden = hidden_rnn
        ##############################################################################
        features = torch.cat([output_new, output_meta], dim=-1)

        return features, hidden

    def _base_init(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, neural_networks.RNN_FEATURES,
                           device=self.fc_meta.weight.device.type)  # create inital state for rnn layer only

    def init_state(self, batch_size: int = 1) -> torch.Tensor:  # model init state
        if self.rnn_type == neural_networks.RNNType.GRU:
            state = self._base_init(batch_size)
        else:
            state = (self._base_init(batch_size), self._base_init(batch_size))

        if self.enable_tn:  # if TimeNormalization is enable then update init state
            state = (state, self.tn.init_state(batch_size=batch_size))
        return state
