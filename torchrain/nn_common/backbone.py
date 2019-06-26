import torch
import torch.nn as nn
from torchrain import nn_common


class Backbone(nn.Module):
    def __init__(self, n_layers: nn_common.NLayers, rnn_type: nn_common.RNNType,
                 enable_tn: bool = False, tn_alpha: float = 0.9, tn_affine: bool = False, tn_indicator_func=None):
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
        self.enable_tn = enable_tn
        if enable_tn:
            self.tn = nn_common.TimeNormalization(alpha=tn_alpha, num_features=nn_common.DYNAMIC_INPUT_SIZE,
                                                  affine=tn_affine)
            self.tn_indicator_func = tn_indicator_func
        self.fc_meta = nn.Linear(nn_common.STATIC_INPUT_SIZE, nn_common.FC_FEATURES)
        self.relu = nn.ReLU()

    def forward(self, input_tensor, input_meta_tensor, hidden):  # model forward pass
        if self.enable_tn:  # split hidden state for RE
            hidden_tn = hidden[2:, :, :]
            hidden_rnn = hidden[:2, :, :]

        output_meta = self.relu(self.fc_meta(input_meta_tensor))
        output, hidden_rnn = self.rnn(input_tensor, hidden_rnn)
        output = output.contiguous()
        output_meta = output_meta.view(output_meta.size()[0], 1, output_meta.size()[1]).repeat(1, output.size()[1], 1)
        ##############################################################################
        if self.enable_tn:  # run TimeNormalization over rnn output and update the state of Backbone
            ind = None
            if self.tn_indicator_func is not None:
                ind = self.tn_indicator_func(torch.cat([output, output_meta], dim=-1))
            output_new, hidden_tn = self.tn(output, hidden_tn, indecator=ind)
            hidden = torch.cat([hidden_rnn, hidden_tn], dim=0)
        else:  # pass rnn state and output
            output_new = output
            hidden = hidden_rnn
        ##############################################################################
        features = torch.cat([output_new, output_meta], dim=-1)

        return features, hidden

    def init_state(self, batch_size=1):  # model init state
        state = torch.zeros(int(self.n_layers.value), batch_size, nn_common.RNN_FEATURES,
                            device=self.config.device)  # create inital state for rnn layer only
        if self.enable_tn:  # if TimeNormalization is enable then update init state
            state = torch.cat([state, self.tn.init_state(batch_size=batch_size)])
        return state
