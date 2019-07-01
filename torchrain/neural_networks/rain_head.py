import torch.nn as nn


class RainHead(nn.Module):
    def __init__(self, n_features: int):
        super(RainHead, self).__init__()
        self.fc = nn.Linear(n_features, 1)

    def forward(self, input_tensor):  # model forward pass
        return self.fc(input_tensor)
