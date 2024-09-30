import torch.nn as nn
import config
import torch
import numpy as np
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.d_model,config.d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(config.d_ff,config.d_model,bias=False),
        )
    def forward(self, inputs):
        residual=inputs
        output=self.fc(inputs)
        return nn.LayerNorm(config.d_model)(output + residual)
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q,K,V,attn_mask):
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(config.d_k)
        scores.masked_fill_(attn_mask,-1e9)
        attn=nn.Softmax(dim=-1)(scores)
        context=torch.matmul(attn,V)
        return context,attn
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q=nn.Linear(config.d_model,config.d_k*config.n_heads,bias=False)
        self.W_K = nn.Linear(config.d_model, config.d_k * config.n_heads, bias=False)
        self.W_V = nn.Linear(config.d_model, config.d_v * config.n_heads, bias=False)
        self.fc=nn.Linear(config.d_v * config.n_heads, config.d_model,bias=False)
    def forward(self,input_Q,input_K,input_V,attn_mask):
        residual,batch_size=input_Q,input_Q.size(0)
        #(B,S,D)=>(B,S,D_NEW)=>(B,S,H,W)=>(B,H,S,W)
        Q=self.W_Q(input_Q).view(batch_size,-1,config.n_heads,config.d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, config.n_heads, config.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, config.n_heads, config.d_v).transpose(1, 2)
        #[batch_size,n_heads,len_k,len_v]
        attn_mask=attn_mask.unsqueeze(1).repeat(1,config.n_heads,1,1)

        context,attn=ScaledDotProductAttention()(Q,K,V,attn_mask)
        # [batch_size,n_heads,len_k,d_v]
        context=context.transpose(1,2).reshape(batch_size,-1,config.n_heads*config.d_v)
        #  [batch_size,len_k,d_model]
        output=self.fc(context)
        return nn.LayerNorm(config.d_model)(output+residual),attn