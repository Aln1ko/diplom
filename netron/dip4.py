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

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = (7,7), stride = (2,2), padding=(3, 3)),# 185, 613
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),

            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),# 185, 613
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),

            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),#93, 307
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),# 93, 307
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),

            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),#47, 154
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#47, 154
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),

            nn.Conv2d(128,256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),# 24, 77
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),# 24, 77
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),

            nn.Conv2d(256,512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),# 12, 39
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),# 12, 39
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  #  6 × 20
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  #  3 × 10
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.AdaptiveAvgPool2d((5, 5))  # Глобальний пулінг до фіксованого розміру
        )
        self.lstm1 = nn.LSTMCell(input_size = 512 * 3 * 10, hidden_size=128)
        self.lstm2 = nn.LSTMCell(input_size = 512 * 3 * 10, hidden_size=128)

        self.network = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self,image1,image2):
        out_features1 = self.features(image1)
        out_features2 = self.features(image2)
        out_features1 = torch.flatten(out_features1, 1)
        out_features2 = torch.flatten(out_features2, 1)
        batch_size = out_features1.size(0)

        # Ініціалізуємо приховані стани та стани осередків для lstm1
        h_t1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=out_features1.device)
        c_t1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=out_features1.device)

        # Пропускаємо ознаки першої картинки через lstm1 (один часовий крок)
        h_t1_next, c_t1_next = self.lstm1(out_features1, (h_t1, c_t1))
        h_t2_next, c_t2_next = self.lstm2(out_features2, (h_t1_next, c_t1_next))
        out = self.network(h_t2_next) # [batch_size, 1]
        return out
