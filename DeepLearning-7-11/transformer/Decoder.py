import torch
import numpy as np
import torch.nn as nn

import submodel
import utils
import config
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn=submodel.MultiHeadAttention()
        self.dec_enc_attn=submodel.MultiHeadAttention()
        self.pos_ffn = submodel.PoswiseFeedForwardNet()
    def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        dec_outputs,dec_self_attn=self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        dec_outputs,dec_enc_attn=self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs,dec_enc_attn_mask)
        #  [batch_size,len_k,d_model]
        dec_outputs=self.pos_ffn(dec_outputs)
        return dec_outputs,dec_self_attn,dec_enc_attn
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(config.tgt_vocab_size,config.d_model)
        self.pos_emb = nn.Embedding.from_pretrained(utils.get_sinusoid_encoding_table(config.tgt_vocab_size,config.d_model),freeze=True)
        self.layers=nn.ModuleList([DecoderLayer() for _ in range(config.n_layers)])
    def forward(self, dec_inputs,enc_inputs,enc_outputs):
        word_emb = self.tgt_emb(dec_inputs)
        pos_emb = self.pos_emb(dec_inputs)
        dec_outputs=word_emb+pos_emb
        dec_self_attn_pad_mask=utils.get_attn_pad_mask(dec_inputs,dec_inputs)
        dec_self_attn_subsequence_mask=utils.get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask+dec_self_attn_subsequence_mask),0)

        dec_enc_attn_mask=utils.get_attn_pad_mask(dec_inputs,enc_inputs)

        dec_self_attns,dec_enc_attns=[],[]
        for layer in self.layers:
            dec_outputs,dec_self_attn,dec_enc_attn=layer(dec_outputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs,dec_self_attns,dec_enc_attns

