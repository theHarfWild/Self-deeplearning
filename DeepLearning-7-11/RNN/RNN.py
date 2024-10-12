import numpy as np
import torch
import torch.nn as nn
import torch.optim as op
import torch.utils.data as data
import Dataloader
import RNNLayer
class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = RNNLayer.RNNLayer(input_size,30,10)
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        x = self.rnn(x)
        x = self.fc(x)
        return x
def loss(pred, target):
    return ((pred - target)**2).sum()
