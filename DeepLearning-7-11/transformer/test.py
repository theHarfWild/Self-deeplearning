import torch.nn as nn
import torch
b=torch.Tensor([
    [[10,11],
     [12,13]],
    [[14,15],
     [16,17]],
    [[18,19],
     [20,21]],
    [[22,23],
     [24,25]]
])
a=b.transpose(-1,-3)
print(b[0,1,1])
print(a[1,1,0])