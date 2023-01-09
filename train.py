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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from functional import *

# %% parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_path = "./data/train2014/"
style_path = "./data/animal-painting/"
train_tf = train_transform(size = (256,256))
content_dataset = FlatFolderDataset(content_path, train_tf)
style_dataset = FlatFolderDataset(style_path, train_tf)

save_dir = "./experiments/"
lr = 1e-4
lr_decay = 5e-5
max_iter = 20000
batch_size = 24
style_weight = 10
content_weight = 1
save_model_interval = 500


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
#trained_state_dict = torch.load("./models/decoder.pth")
# 1. filter out unnecessary keys
#trained_state_dict = {k: v for k, v in trained_state_dict.items() if k in decoder.state_dict()}
# 2. overwrite entries in the existing state dict
#decoder.state_dict().update(trained_state_dict) 

decoder.load_state_dict(torch.load("experiments/decoder_iter_8750_loss_7.64.pth.tar"))

net = StyleTransferNet(encoder, decoder)
net.to(device)

content_iter = iter(data.DataLoader(content_dataset, batch_size, sampler=InfiniteSamplerWrapper(content_dataset)))
style_iter = iter(data.DataLoader(style_dataset, batch_size, sampler=InfiniteSamplerWrapper(style_dataset)))

# %% Training loop
optimizer = torch.optim.Adam(net.decoder.parameters(), lr=lr)

loss_content = []
loss_style = []

for i in range(8750, max_iter):
    # if i % scheduler["reset_every"] == 0:
    #     lr = lr * scheduler["scale"]
    #     reset_lr(optimizer, lr)
    reset_lr(optimizer, lr / (1 + lr_decay * i))

    alpha = np.random.choice([1, 0.9, 0.8, 0.7, 0.5])
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    if content_images.shape != style_images.shape:
        continue
    loss_c, loss_s = net(content_images, style_images, alpha)
    loss_c = content_weight * loss_c
    loss_s = style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_content.append(loss_c.item())
    loss_style.append(loss_s.item())

    if (i + 1 ) % 250 == 0 or i == 0:
        print(f"Iteration {i}, content loss {loss_c.item():.2f}, style loss {loss_s.item():.2f}")
#        with torch.no_grad():
#            output = style_transfer(encoder, decoder, content_images[0:1], content_images[0:1])
#            fig, axs = plt.subplots(1, 3, figsize = (7, 21))
#            axs[0].imshow(content_images[0].detach().cpu().permute(1,2,0).numpy())
#            axs[0].set_title("Content image")

#            axs[1].imshow(style_images[0].detach().cpu().permute(1,2,0).numpy())
#            axs[1].set_title("Style image")

#            axs[2].imshow(output[0].detach().cpu().permute(1,2,0).numpy())
#            axs[2].set_title("Result image")
#            plt.show()


    if (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        
        torch.save(state_dict, os.path.join(save_dir, f'decoder_iter_{i + 1}_loss_{loss.item():.2f}.pth.tar'))
