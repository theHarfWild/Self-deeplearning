import torch
import torch.nn as nn
import config
from Discriminator import Discriminator
from Generator import Generator
generator=Generator()
discriminator=Discriminator()
g_opt=torch.optim.Adam(generator.parameters(),lr=1e-4)
d_opt=torch.optim.Adam(discriminator.parameters(),lr=1e-4)

loss_fn=nn.BCELoss()

epochs=40

for epoch in range(epochs):
    for mini_batch in dataloader:
        gt_images,_=mini_batch
        z=torch.randn(config.batch_size,config.rand_dim)
        pred_images=generator(z)
        target=torch.ones(config.batch_size,1)
        g_loss=loss_fn(discriminator(pred_images),target)
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        d_opt.zero_grad()
        d_loss=0.5*(loss_fn(discriminator(gt_images),torch.ones(config.batch_size,1))+loss_fn(discriminator(pred_images.detach(),torch.zeros(config.batch_size,1))))
        d_loss.backward()
        d_opt.step()