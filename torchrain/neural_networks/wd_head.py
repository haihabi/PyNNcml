import torch.nn as nn


class WetDryHead(nn.Module):
    def __init__(self, n_features: int):
        super(WetDryHead, self).__init__()
        self.fc = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):  # model forward pass
        return self.sigmoid(self.fc(input_tensor))
