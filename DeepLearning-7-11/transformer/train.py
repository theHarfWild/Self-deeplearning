import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from transformer import Transformer
#
import Dataloader
#1.数据集部分
sentences=[
    ['ich mochte ein bier P','S i want a beer .','i want a beer . E'],
    ['ich mochte ein cola P','S i want a coke .','i want a coke . E'],
]
#原语库
src_vocab={'P':0,'ich':1,'mochte':2,'ein':3,'bier':4,'cola':5}
#src_vocab_size=len(src_vocab)
#目标语库
tgt_vocab={'S':0,'i':1,'want':2,'a':3,'beer':4,'coke':5,'.':6,'E':7,'P':8}
#tgt_vocab_size=len(tgt_vocab)
idx2word={i:w for i,w in enumerate(tgt_vocab)}
enc_inputs, dec_inputs, dec_outputs = Dataloader.make_data(sentences,src_vocab,tgt_vocab)
#实例化数据加载类
loader=Data.DataLoader(Dataloader.MyDataset(enc_inputs, dec_inputs, dec_outputs),2,True)

#2.模型构建部分
model=Transformer()
criterion=nn.CrossEntropyLoss(ignore_index=0)
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.99)
for epoch in range(30):
    for enc_inputs,dec_inputs,dec_outputs in loader:
        outputs,enc_self_attns,dec_self_attns,dec_enc_attns=model(enc_inputs,dec_inputs)
        loss=criterion(outputs,dec_outputs.view(-1))
        print(f'Epoch: {epoch+1}',f'loss = {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print(enc_inputs,dec_inputs[:,2:3])
print(dec_inputs)
result,a,b,c=model(enc_inputs,dec_inputs[:,2:3])
print(result)
lis=result.argmax(1).tolist()
for i in lis:
    print(idx2word[i])