import numpy as np
import torch
import torch.nn as nn
class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.memorylayer = nn.Linear(hidden_size, hidden_size)
        self.hiddenlayer = nn.Linear(hidden_size, output_size)
        self.inputlayer=nn.Linear(input_size, hidden_size)
    def forward(self, x):#x:(b,s,l)
        out=torch.zeros(*x.shape[0:2],self.hidden_size)
        for i in range(x.shape[1]):
            self.memory=None
            insq=x[:,i,:]
            if self.memory is None:
                self.memory = self.inputlayer(insq)
            else:
                memo = self.memorylayer(self.memory)
                self.memory = memo + self.inputlayer(x)
            out[:,i,:]=self.memory
            output = self.hiddenlayer(nn.ReLU()(out))
            return output
