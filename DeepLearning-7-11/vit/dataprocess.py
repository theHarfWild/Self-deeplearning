import torch
import torch.utils.data as data
import numpy as np
import config
class MyTrainDataset(data.Dataset):
    def __init__(self):
        super(MyTrainDataset, self).__init__()
        self.data1=np.load('..\\cifar-10-batches-py\\data_batch_1', encoding='bytes', allow_pickle=True)
        self.data1=self.data1[b'data'],self.data1[b'labels']
        self.data2=np.load('..\\cifar-10-batches-py\\data_batch_2', encoding='bytes', allow_pickle=True)
        self.data2 = self.data2[b'data'], self.data2[b'labels']
        self.data3= np.load('..\\cifar-10-batches-py\\data_batch_3', encoding='bytes', allow_pickle=True)
        self.data3 = self.data3[b'data'], self.data3[b'labels']
        self.data4= np.load('..\\cifar-10-batches-py\\data_batch_4', encoding='bytes', allow_pickle=True)
        self.data4 = self.data4[b'data'], self.data4[b'labels']
        self.data5= np.load('..\\cifar-10-batches-py\\data_batch_5', encoding='bytes', allow_pickle=True)
        self.data5 = self.data5[b'data'], self.data5[b'labels']
        self.data=np.concatenate((self.data1[0],self.data2[0],self.data3[0],self.data4[0],self.data5[0]),axis=0)
        self.data=torch.tensor(self.data,dtype=torch.float).view(50000,3,32,32)
        self.labels = torch.tensor(np.concatenate((self.data1[1], self.data2[1], self.data3[1], self.data4[1], self.data5[1]), axis=0),dtype=torch.long)
    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    def __len__(self):
        return self.data.shape[0]//config.batch_size*config.batch_size
traindataloader=data.DataLoader(MyTrainDataset(),batch_size=config.batch_size,shuffle=True)
class MyTestDataset(data.Dataset):
    def __init__(self):
        super(MyTestDataset, self).__init__()
        self.data=np.load('..\\cifar-10-batches-py\\test_batch', encoding='bytes', allow_pickle=True)
        self.labels,self.data = self.data[b'labels'],self.data[b'data']
        self.data = torch.tensor(self.data,dtype=torch.float).view(10000, 3, 32, 32)
        self.labels=torch.tensor(self.labels,dtype=torch.long)
    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    def __len__(self):
        return self.data.shape[0]//config.batch_size*config.batch_size
testdataloader=data.DataLoader(MyTestDataset(),batch_size=config.batch_size,shuffle=True)