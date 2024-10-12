import numpy as np
import torch
import torch.nn as nn
import torch.optim as op
import torch.utils.data as data
import Dataloader
import RNN
myset=Dataloader.MyDataset(Dataloader.datas)
loader=data.DataLoader(myset, batch_size=10, shuffle=True)
mod=RNN.RNN(1)
opt=op.Adam(mod.parameters(), lr=0.001)
for epoch in range(500):
    los=0
    for data,result in loader:
        re=mod(data)
        loss=RNN.loss(re,result)
        los+=loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print('Epoch:',epoch,'Loss:',los)
    a=torch.tensor([[[46.],[47.],[48.]]])
print(mod(a))