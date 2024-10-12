import numpy as np
import torch
import torch.nn as nn
import torch.optim as op
import torch.utils.data as data
from RNNLayer import RNNLayer
a=torch.tensor([[1,2,3],[4,5,6],[7,8,11]])
b=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(((a-b)**2).sum())