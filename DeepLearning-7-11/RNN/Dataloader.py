import torch.utils.data as data
import numpy as np
import torch
datas=[]
for i in range(100):
    u=[]
    for j in range(4):
        u.append([i+j])
    datas.append(u)
datas=np.array(datas)
class MyDataset(data.Dataset):
    def __init__(self, datas):
        self.datas=torch.tensor(datas).type(torch.FloatTensor)
    def __getitem__(self, index):
        return self.datas[index,0:3],self.datas[index,1:]
    def __len__(self):
        return self.datas.shape[0]