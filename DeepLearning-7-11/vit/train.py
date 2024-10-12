import torch
import VisisonTransformer
import dataprocess
import numpy as np
import os
import matplotlib.pyplot as plt
def load(path):
    if os.path.exists(path):
        model = torch.load(path)
        print('model has been loaded')
    else:
        model = VisisonTransformer.VisionTransformer()
        print('newed a model')
    return model
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainset=dataprocess.traindataloader
testset=dataprocess.testdataloader
#是否使用之前的
model=load('vitsmodel\\VisionTransformer29.pt')
opt=torch.optim.AdamW(model.parameters(), lr=3e-5)
loss=torch.nn.CrossEntropyLoss()
model=model.cuda()
model.check_device()

#记录acc数据
ontrain=[]
ontest=[]
#开始训练
try:
    for epoch in range(30):
        los = 0
        acc = []
        model.train()
        for images, labels in trainset:
            images = images.to(device)
            labels = labels.to(device)
            pre = model.forward(images)
            lossnum = loss(pre, labels)
            opt.zero_grad()
            lossnum.backward()
            opt.step()
            los += lossnum
            presu = pre.argmax(dim=1, keepdim=True)
            a = sum([1 if presu[i] == labels[i] else 0 for i in range(presu.size(0))]) / len(labels)
            acc.append(a)
        torch.save(model, f'vitsmodel\\VisionTransformer{epoch}.pt')
        acc = np.mean(acc)
        print(f"epoch {epoch} loss {los} acc {acc},model saved")
        ontrain.append(acc)
        # 测试集准确率
        model.eval()
        evalacc = []
        for images, labels in testset:
            images = images.to(device)
            labels = labels.to(device)
            pre = model.forward(images)
            pre = pre.argmax(dim=1, keepdim=True)
            a = sum([1 if pre[i] == labels[i] else 0 for i in range(pre.size(0))]) / len(labels)
            evalacc.append(a)
        evalacc = np.mean(evalacc)
        print(f"epoch {epoch} eval acc {evalacc}")
        ontest.append(acc)
finally:
    plt.subplot(1,2,1)
    x=np.array(list((i for i in range(len(ontrain)))))
    plt.plot(x, ontrain, ls="-.", color="r", marker=",", lw=2, label="trainacc")
    plt.subplot(1,2,2)
    x=np.array(list((i for i in range(len(ontest)))))
    plt.plot(x,ontest, ls="-.", color="r", marker=",", lw=2, label="testacc")
    plt.show()


