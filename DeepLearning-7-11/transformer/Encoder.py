import torch
import numpy as np
import torch.nn as nn

import config
import utils
from submodel import MultiHeadAttention
from submodel import PoswiseFeedForwardNet
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn=MultiHeadAttention()
        self.pos_ffn=PoswiseFeedForwardNet()
    def forward(self,enc_inputs,enc_self_attn_mask):
        #自注意头
        enc_outs,attn=self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        #前馈网络
        enc_outputs=self.pos_ffn(enc_outs)
        return enc_outputs,attn
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sec_emb=nn.Embedding(config.src_vocab_size,config.d_model)
        self.pos_emb=nn.Embedding.from_pretrained(utils.get_sinusoid_encoding_table(config.src_vocab_size,config.d_model),freeze=True)
        self.layers=nn.ModuleList([EncoderLayer() for _ in range(config.n_layers)])

    def forward(self, enc_inputs):
        word_emb = self.sec_emb(enc_inputs)
        pos_emb = self.pos_emb(enc_inputs)
        #编码完成
        enc_outputs=word_emb+pos_emb
        #获得pmask
        enc_self_attn_mask=utils.get_attn_pad_mask(enc_inputs,enc_inputs)
        enc_self_attns=[]
        for layer in self.layers:
            #每层传入编码和mask
            enc_outputs,enc_self_attn=layer(enc_outputs,enc_self_attn_mask)
            #得到每一层的enc_self_attn和output
            enc_self_attns.append(enc_self_attn)
        #返回最终输出结果和注意力表
        return enc_outputs,enc_self_attns

