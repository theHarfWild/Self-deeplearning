import numpy as np
import torch
def get_sinusoid_encoding_table(n_position,d_model):
    def cal_angle(position,hid_idx):
        return position/np.power(10000, 2*(hid_idx//2)/d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position,hid_j) for hid_j in range(d_model)]
    sinusoid_table=np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:,0::2]=np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:,1::2]=np.cos(sinusoid_table[:,1::2])
    return torch.FloatTensor(sinusoid_table)
def get_attn_pad_mask(seq_q,seq_k):
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size,len_q,len_k)
def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0),seq.size(1),seq.size(1)]
    subsequence_mask=np.triu(np.ones(attn_shape),k=1)
    subsequence_mask=torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask
