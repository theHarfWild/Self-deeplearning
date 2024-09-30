import torch.utils.data as Data
import torch

def make_data(sentences,src_vocab,tgt_vocab):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        #转换成序号
        enc_input=[[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input=[[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output=[[tgt_vocab[n] for n in sentences[i][2].split()]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    #得到[[序号1，序号2],[]]的形式
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
#转化成LongTensor二维数组,其中第一维为句子，第二维单词
class MyDataset(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataset, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    def __len__(self):
        return self.enc_inputs.shape[0]
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
