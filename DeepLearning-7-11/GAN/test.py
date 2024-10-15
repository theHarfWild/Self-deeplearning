import numpy as np
import torch
import torch.nn as nn
import config
from Discriminator import Discriminator
from Generator import Generator
from dataprocess import dataloader
import os
import matplotlib.pyplot as plt
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator=torch.load('generator.pt')
generator=generator.to(device)
generator.eval()
z=torch.randn(9,config.rand_dim).to(device)
res=generator(z).view(9,28,-1)
for i in range(9):
    plt.subplot(3,3,i+1)
    a=np.array(res[i,:,:].detach().cpu())
    plt.imshow(a/a.max()*255,cmap='gray')
plt.show()