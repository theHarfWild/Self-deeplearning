import torch
import torch.nn as nn
import config
class Generator(nn.Module):
    def __init__(self,in_dim):
        super(Generator,self).__init__()
        self.model= nn.Sequential(
            nn.Linear(in_dim,64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, config.data_size),
            nn.Tanh(),
        )
    def forward(self,x):
        output=self.model(x)
        return output
        pass
