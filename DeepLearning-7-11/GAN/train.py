import torch
import torch.nn as nn
import config
from Discriminator import Discriminator
from Generator import Generator
from dataprocess import dataloader
import os
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if os.path.exists('generator.pt'):
    generator=torch.load('generator.pt')
else:
    generator=Generator(config.rand_dim)
if os.path.exists('discriminator.pt'):
    discriminator=torch.load('discriminator.pt')
else:
    discriminator=Discriminator()
g_opt=torch.optim.Adam(generator.parameters(),lr=5e-5)
d_opt=torch.optim.Adam(discriminator.parameters(),lr=1e-5)
generator=generator.to(device)
discriminator=discriminator.to(device)
loss_fn=nn.BCELoss()


epochs=100

for epoch in range(epochs):
    for mini_batch in dataloader:
        mini_batch=mini_batch.to(device)
        gt_images=mini_batch
        z=torch.randn(config.batch_size,config.rand_dim).to(device)
        pred_images=generator(z)
        target=torch.ones(config.batch_size,1).to(device)
        g_loss=loss_fn(discriminator(pred_images),target)
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        d_opt.zero_grad()
        d_loss=0.3*(loss_fn(discriminator(gt_images),torch.ones(config.batch_size,1).to(device))+loss_fn(discriminator(pred_images.detach()),torch.zeros(config.batch_size,1).to(device)))
        d_loss.backward()
        d_opt.step()
    print(f'epoch {epoch} gloss {g_loss.item()},dloss {d_loss.item()}')
    torch.save(generator,'generator.pt')
    torch.save(discriminator,'discriminator.pt')