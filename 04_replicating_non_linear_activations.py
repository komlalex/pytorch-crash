"""repliting non-linear activation functions 
Neural networks rather than us telling the model what to learn, we can give it 
tools to discover patterns in the data on its own
And these tools are linear & non-linear functions

"""
import torch 
from torch import nn 
import matplotlib.pyplot as plt 
import numpy as np 

# Create a tensor 
A = torch.arange(-10, 10, 1, dtype=torch.float32)  

# ReLU 
def relu(x: torch.Tensor) -> torch.Tensor: 
    return torch.maximum(torch.tensor(0), x)

plt.plot(relu(A)) 
plt.show()