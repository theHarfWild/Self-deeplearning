import torch
import torch.nn as nn
import config
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.c=torch.zeros(config.csize)
        self.h=torch.zeros(config.hsize)
        self.whf=nn.Linear(config.hsize,config.fsize,bias=True)
        self.whi=nn.Linear(config.fsize,config.isize,bias=True)
        self.whg=nn.Linear(config.isize,config.gsize,bias=True)
        self.who=nn.Linear(config.hsize,config.osize,bias=True)
        self.wxf=nn.Linear(config.insize,config.fsize,bias=True)
        self.wxi=nn.Linear(config.innsize,config.isize,bias=True)
        self.wxg=nn.Linear(config.innsize,config.gsize,bias=True)
        self.wxo=nn.Linear(config.gsize,config.osize,bias=True)
    def forward(self, x):#（batch,sq,ds）
        for i in range(x.shape[1]):
            nowx=x[:,i,:]
            f = self.sigmoid(self.whf(self.h) + self.wxf(nowx))
            i = self.sigmoid(self.whi(self.h) + self.wxi(nowx))
            g = self.tanh(self.whg(self.h) + self.wxg(nowx))
            o = self.sigmoid(self.who(self.h) + self.wxo(nowx))
            self.c = self.c * f + i * g
            self.h = self.tanh(self.c) * o
        return self.h
