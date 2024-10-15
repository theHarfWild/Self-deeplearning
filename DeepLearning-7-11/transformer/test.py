import torch.nn as nn
import torch
#           (卷积核厚度，  卷积核个数       ，卷积核面积）
c=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1,bias=True,groups=3)
for name, parameters in c.named_parameters():
    print(name, ':', parameters.size())