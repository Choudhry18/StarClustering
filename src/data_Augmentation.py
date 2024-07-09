
Copy code
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_du
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle
from starcnet import Net
import torch.nn.init as init

# Custom scaling transformation
class RandomScale:
    def __init__(self, scale_factor=1.07, probability=0.5):
        self.scale_factor = scale_factor
        self.probability = probability

    def __call__(self, x):
        if np.random.rand() < self.probability:
            scale = self.scale_factor
            size = (int(x.shape[1] * scale), int(x.shape[2] * scale))
            x = transforms.Resize(size)(x)
            x = transforms.CenterCrop((32, 32))(x)  # Assuming original size is 32x32
        return x

# Data augmentation transforms
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomScale(scale_factor=1.07, probability=0.5),
    transforms.RandomRotation([90, 270])
])
