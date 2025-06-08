import os
import os.path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms
import math
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights

class CNNBackbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base = resnet18(weights=weights)

        new_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight[:] = base.conv1.weight.mean(dim=1, keepdim=True)[:, :2, :, :]  # среднее по каналам и обрезка до 2

        base.conv1 = new_conv1

        self.encoder = nn.Sequential(*list(base.children())[:-2]) # without avgpool and fc
        self.conv_proj = nn.Conv2d(512, out_dim, kernel_size=1)    # [B, 512, H/32, W/32] -> [B, out_dim, H/32, W/32]

    def forward(self, x):
        x = self.encoder(x)     # [B, 512, H/32, W/32]
        x = self.conv_proj(x)   # [B, out_dim, H/32, W/32]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class Network(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.backbone = CNNBackbone(out_dim=feature_dim)
        self.flatten = nn.Flatten(2)  # [B, C, H, W] -> [B, C, H*W]
        self.transpose = lambda x: x.permute(0, 2, 1)  # -> [B, H*W, C]
        self.pos_embed = PositionalEncoding(d_model=feature_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, N, D] -> [B, D]
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # (tx, ty, tz, rx, ry, rz)
        )

    def forward(self, image1,image2):
        x = torch.cat((image1,image2),dim = 1)
        B, C, H, W = x.shape  # x: [B, 2, H, W] for 2 RGB images -> stacked
        feat = self.backbone(x)  # [B, D, H', W']
        feat = self.flatten(feat)
        feat = self.transpose(feat)
        feat = self.pos_embed(feat)  # [B, N, D]
        feat = self.transformer(feat)  # [B, N, D]
        pooled = feat.mean(dim=1)  # or self.pool(feat.transpose(1, 2)).squeeze(-1)
        out = self.regressor(pooled)
        return out
