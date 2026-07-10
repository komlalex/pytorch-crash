"""
Computer vision librariesi in PyTorch 
* torchvision 
* torchvision.datasets 
* torchvision.models 
* torchvision.transforms
* torch.utils.data.Dataset 
* torch.utils.data.DataLoader""" 
# Import torch
import torch    
from torch import nn  

# Import torchvision 
import torchvision
from torchvision import datasets 
from torchvision import transforms 
from torchvision.transforms import ToTensor

# Import matplotlib 
import matplotlib.pyplot as plt  
print(torchvision.__version__)

"""Get the dataset 
The dataset used is fashion MNIST from torchvision.datasets""" 

# Setup training data 
train_data = datasets.FashionMNIST(
    root="data", 
    train=True, 
    download=True, 
    transform=ToTensor(),
    target_transform=None
) 

train_data = datasets.FashionMNIST(
    root="data", 
    train=False, 
    download=True, 
    transform=ToTensor(),
    target_transform=None
)