import torch
import torch.nn as nn

class Pre_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Pre_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, 128 *4)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 4, 128)
        return x
