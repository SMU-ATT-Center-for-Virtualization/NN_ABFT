import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from injector import ModelInjector


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)