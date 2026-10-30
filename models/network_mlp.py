import torch.nn as nn


class TrafficMLP(nn.Module):
    def __init__(self, input_dim=6, hidden=(64, 32), dropout=0.05):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)