import numpy as np
import torch
import torch.nn as nn
import config
# 输入带batch的图片，(batch,H,W,C)
# 输出embedding后的图片向量(batch,N,P^2*C)
# 向量经过FC (batch,N,dim)
class PatchNet(nn.Module):
    def check_device(self):
        for name, param in self.named_parameters():
            if param.device != torch.device('cuda:0'):
                print(f"Parameter '{name}' is on {param.device}")
        
        for i in self.lis:
            i.check_device()
    def __init__(self):
        super(PatchNet, self).__init__()
        self.H=config.H
        self.W=config.W
        self.C=config.C
        self.P=config.P
        self.D=config.d_model
        self.H1=self.H//self.P
        self.W1=self.W//self.P
        self.pos_embed = nn.Parameter(torch.zeros(1, self.H1*self.W1+1, self.D))
        self.cls_token = nn.Parameter(torch.zeros(config.batch_size, 1, self.D))
        self.embedding = nn.Linear(self.P**2*self.C,self.D)
        self.lis=[]

    def forward(self, batched_images):
        #将图像处理成正好可被分割
        batched_images = batched_images[:,0:self.H1*self.P,0:self.W1*self.P,:]
        batched_images = batched_images.view(batched_images.size(0),self.H1*self.W1,self.C*self.P**2)
        #分割完转变为向量
        batched_images = self.embedding(batched_images)
        #加入分类头
        batched_images = torch.cat((batched_images,self.cls_token),dim=1)
        #加入posembedding
        out_embed=self.pos_embed+batched_images
        return out_embed
class PoswiseFeedForwardNet(nn.Module):
    def check_device(self):
        for name, param in self.named_parameters():
            if param.device != torch.device('cuda:0'):
                print(f"Parameter '{name}' is on {param.device}")
        
        for i in self.lis:
            i.check_device()
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.d_model,config.d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(config.d_ff,config.d_model,bias=False),
        )
        self.lnorm = nn.LayerNorm(config.d_model)
        self.lis=[]
    def forward(self, inputs):
        residual=inputs
        output=self.fc(inputs)
        return self.lnorm(output + residual)
class ScaledDotProductAttention(nn.Module):
    def check_device(self):
        for name, param in self.named_parameters():
            if param.device != torch.device('cuda:0'):
                print(f"Parameter '{name}' is on {param.device}")
        
        for i in self.lis:
            i.check_device()
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.lis=[]
    def forward(self, Q,K,V):
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(config.d_k)
        attn=nn.Softmax(dim=-1)(scores)
        context=torch.matmul(attn,V)
        return context
class MultiHeadAttention(nn.Module):
    def check_device(self):
        for name, param in self.named_parameters():
            if param.device != torch.device('cuda:0'):
                print(f"Parameter '{name}' is on {param.device}")
        
        for i in self.lis:
            i.check_device()
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q=nn.Linear(config.d_model,config.d_k*config.n_heads,bias=False)
        self.W_K = nn.Linear(config.d_model, config.d_k * config.n_heads, bias=False)
        self.W_V = nn.Linear(config.d_model, config.d_v * config.n_heads, bias=False)
        self.fc=nn.Linear(config.d_v * config.n_heads, config.d_model,bias=False)
        self.lnorm=nn.LayerNorm(config.d_model)
        self.lis=[]
    def forward(self,input_Q,input_K,input_V):
        residual,batch_size=input_Q,input_Q.size(0)
        #(B,S,D)=>(B,S,D_NEW)=>(B,S,H,W)=>(B,H,S,W)
        Q=self.W_Q(input_Q).view(batch_size,-1,config.n_heads,config.d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, config.n_heads, config.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, config.n_heads, config.d_v).transpose(1, 2)
        #[batch_size,n_heads,len_k,len_v]
        context=ScaledDotProductAttention()(Q,K,V)
        # [batch_size,n_heads,len_k,d_v]
        context=context.transpose(1,2).reshape(batch_size,-1,config.n_heads*config.d_v)
        #  [batch_size,len_k,d_model]
        output=self.fc(context)
        return self.lnorm(output+residual)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()