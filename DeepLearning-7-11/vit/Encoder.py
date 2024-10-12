import torch.nn as nn
import Submodel
import config
import torch
class EncoderLayer(nn.Module):
    def check_device(self):
        for name, param in self.named_parameters():
            if param.device != torch.device('cuda:0'):
                print(f"Parameter '{name}' is on {param.device}")
        
        for i in self.lis:
            i.check_device()
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn=Submodel.MultiHeadAttention()
        self.pos_ffn=Submodel.PoswiseFeedForwardNet()
        self.lis=[self.enc_self_attn,self.pos_ffn]
    def forward(self,enc_inputs):
        #自注意头
        enc_outs=self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs)
        #前馈网络
        enc_outputs=self.pos_ffn(enc_outs)
        return enc_outputs
class Encoder(nn.Module):
    def check_device(self):
        for name, param in self.named_parameters():
            if param.device != torch.device('cuda:0'):
                print(f"Parameter '{name}' is on {param.device}")
        
        for i in self.lis:
            i.check_device()
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers=nn.ModuleList([EncoderLayer() for _ in range(config.n_layers)])
        self.lis=self.layers

    def forward(self, enc_inputs):
        # word_emb = self.sec_emb(enc_inputs)
        # pos_emb = self.pos_emb(enc_inputs)
        # #编码完成
        # enc_outputs=word_emb+pos_emb
        enc_outputs=enc_inputs
        #获得pmask
        enc_self_attns=[]
        for layer in self.layers:
            #每层传入编码和mask
            enc_outputs=layer(enc_outputs)
        #返回最终输出结果和注意力表
        return enc_outputs