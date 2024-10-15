import cv2
import numpy as np
import torch.utils.data as Data
import os
import torch
import config
def loaddata(path):
    if(os.path.exists(path)):
        ar=np.load(path)
    else:
        lis = []
        # for i in os.listdir('..\\mnist\\trainingSet'):
        #     for j in os.listdir('..\\mnist\\trainingSet\\' + i):
        #         path = '..\\mnist\\trainingSet\\' + i + '\\' + j
        #         lis.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        # ar = np.array(lis)
        # np.save('data.npy',ar)
        i='1'
        for j in os.listdir('..\\mnist\\trainingSet\\' + i):
            path = '..\\mnist\\trainingSet\\' + i + '\\' + j
            lis.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        ar = np.array(lis)
        np.save('data.npy',ar)
    return ar
ar=loaddata('data.npy')
class GANMnistdataset(Data.Dataset):
    def __init__(self, ar):
        super(GANMnistdataset, self).__init__()
        self.data = torch.tensor(ar, dtype=torch.float).view(ar.shape[0],-1)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]//config.batch_size*config.batch_size
dataloader=Data.DataLoader(dataset=GANMnistdataset(ar), batch_size=config.batch_size, shuffle=True)