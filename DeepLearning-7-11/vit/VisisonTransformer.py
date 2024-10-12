from torch import nn
from Submodel import PatchNet
from Encoder import Encoder
import config
import torch
class VisionTransformer(nn.Module):
    def check_device(self):
        for name, param in self.named_parameters():
            if param.device != torch.device('cuda:0'):
                print(f"Parameter '{name}' is on {param.device}")
        for i in self.lis:
            i.check_device()
    def __init__(self):
        super(VisionTransformer,self).__init__()
        self.patchembedding=PatchNet()
        self.encoder=Encoder()
        self.fc = nn.Sequential(
            nn.Linear(config.d_model,3*config.d_model),
            nn.Linear(config.d_model*3,config.cls),
        )
        self.sm=nn.Softmax(dim=-1)
        self.lis=[self.patchembedding,self.encoder]
    def forward(self,batched_imgs):
        embedding = self.patchembedding(batched_imgs)
        codered=self.encoder(embedding)
        codered=codered[:,-1,:]
        cls=self.fc(codered)
        return self.sm(cls)