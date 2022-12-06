# %% Import
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 10000000000 
from functional import *

# %% parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_path = "./data/train2014/"
style_path = "./data/test/"
train_tf = train_transform(size = (256,256))
content_dataset = FlatFolderDataset(content_path, train_tf)
style_dataset = FlatFolderDataset(style_path, train_tf)

save_dir = "./experiments"
lr = 1e-4
scheduler = {"reset_every": 10000, "scale": 0.9}
max_iter = 160000
batch_size = 16
style_weight = 10
content_weight = 1
save_model_interval = 10000


encoder = torchvision.models.vgg19(weights = "VGG19_Weights.DEFAULT", progress = True).features
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'), #increase the spatial resolution of the feature maps
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'), #increase the spatial resolution of the feature maps
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 32, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(32, 32, (3, 3)),

    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(32, 16, (3, 3)),

    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(16, 3, (3, 3)),

)
trained_state_dict = torch.load("./models/decoder.pth")
# 1. filter out unnecessary keys
trained_state_dict = {k: v for k, v in trained_state_dict.items() if k in decoder.state_dict()}
# 2. overwrite entries in the existing state dict
decoder.state_dict().update(trained_state_dict) 

net = StyleTransferNet(encoder, decoder)
net.to(device)

content_iter = iter(data.DataLoader(content_dataset, batch_size, shuffle=True))
style_iter = iter(data.DataLoader(style_dataset, batch_size, shuffle=True))

# %% Training loop
optimizer = torch.optim.Adam(net.decoder.parameters(), lr=lr)

loss_content = []
loss_style = []

for i in range(max_iter):
    if i % scheduler["reset_every"] == 0:
        lr = lr * scheduler["scale"]
        reset_lr(optimizer, lr)


    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s = net(content_images, style_images)
    loss_c = content_weight * loss_c
    loss_s = style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_content.append(loss_c.item())
    loss_style.append(loss_s.item())

    if (i + 1 ) % 500 == 0 or i == 0:
        print(f"Iteration {i}, content loss {loss_c.item():.2f}, style loss {loss_s.item():.2f}")


    if (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        
        torch.save(state_dict, save_dir /
                'decoder_iter_{:d}.pth.tar'.format(i + 1))
