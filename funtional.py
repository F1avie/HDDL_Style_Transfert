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
from tqdm import tqdm

# %% Functions
class AdaptiveIN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, content_feat, style_feat):
        return adaptive_instance_normalization(content_feat, style_feat)

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
    
class StyleTransferNet(nn.Module):
    def __init__(self, encoder, decoder, style_loss_index = [4, 9, 16, 23, 32, 37]):
        super(StyleTransferNet, self).__init__()
        # Want to fix the encoder, disable require grad
        for _, param in encoder.named_parameters():
            param.requires_grad = False

        encoder_layers = list(encoder.children())
        self.n_style_checkpoints = len(style_loss_index)
        style_loss_index.insert(0, 0)

        self.style_loss_checkpoints = [
            nn.Sequential(*encoder_layers[style_loss_index[i]:style_loss_index[i+1]])
            for i in range(len(style_loss_index) - 1)
        ]

        self.encoder = encoder
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()        

    # Encode the style image and store the intermediate values for 
    # calculating style loss 
    def encode_with_intermediate(self, input):
        # encode the input image and store the intermediate value for 
        # calculating the final style loss
        results = [input]
        for i in range(self.n_style_checkpoints):
            func = self.style_loss_checkpoints[i]
            results.append(func(results[-1]))
        return results[1:]

    # Encode the input image to get the features
    def encode(self, input):
        for i in range(self.n_style_checkpoints):
            input = self.style_loss_checkpoints[i](input)
        return input

    # Content loss: (encoder(decoder(input)) - input)).norm()
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    # Style loss 
    def calc_style_loss(self, input, target):
        """
        Style loss for a single couple of (input, target) images
        """
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1    # constant to adjust the compromise between content and style
        style_feats = self.encode_with_intermediate(style)   # intermediate values during encoding 
        content_feat = self.encode(content)   # f(c)
        t = adaptive_instance_normalization(content_feat, style_feats[-1])     # t = AdapIN(f(c))
        t = alpha * t + (1 - alpha) * content_feat

        # g(t) - the generated image by the decoder
        g_t = self.decoder(t)     
        # Re-passing the generated image to the encoder for calculating losses  
        g_t_feats = self.encode_with_intermediate(g_t)     

        # Content loss between t = f(c) and f(g(t))
        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        # Style loss during encoding-phase between s and f(g(t))
        # it's the sum over pre-selected layers of encoder
        loss_s = torch.sum(torch.stack([
            self.calc_style_loss(g_t_feats[i], style_feats[i]) for i in range(self.n_style_checkpoints) 
        ]))
        
        return loss_c, loss_s

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def train_transform(size = (256,256)):
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(size=size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def reset_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31