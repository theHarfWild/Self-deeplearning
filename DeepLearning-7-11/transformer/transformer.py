import torch
import numpy as np
import torch.nn as nn

import Dataloader
from Encoder import Encoder
from Decoder import Decoder
import config
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        #  [batch_size,len_k,tgt_vocab_size]
        self.projection = nn.Linear(config.d_model, config.tgt_vocab_size,bias=False)
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs,enc_self_attns=self.encoder(enc_inputs)
        dec_outputs,dec_self_attns,dec_enc_attns=self.decoder(dec_inputs,enc_inputs,enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1,dec_logits.size(-1)),enc_self_attns,dec_self_attns,dec_enc_attns
