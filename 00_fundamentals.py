import torch 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# INTRODUCTION TO TENSORS 
# 1 Creating tnesors 
# Scalar 
# PyTorch Tensors are created using torch.Tensor()
scalar = torch.tensor(2)
#print(scalar.ndim)
#print(scalar.shape)
#print(scalar.dtype)
#print(scalar.item()) # Get tensor back as python data type  

# 2 Vector
# A vector is a one-dimensional tensor 
vector = torch.tensor([7,7])  

# 3 Matrix 
# A matrix is a two-dimensional tensor 
MATRIX = torch.tensor([[7, 8], [9, 10], [11, 12]]) 

# 4 Tensor 
TENSOR =   torch.tensor([[[1, 2, 3], 
                          [4, 5, 6], 
                          [7, 8, 9]]]) 

"""
RANDOM TENSORS 

Random tensors are important because the way many neural networks learn is that they start 
with tensors full of random numbers and then adjust those random numbers 
to better represent the data  
"""
# Create a random tensor  of shape (3, 4) 
random_tensor = torch.rand(3, 4) 


# Create a random tensor with similar shape to an image 
random_image_tensor = torch.rand(size=(224, 224, 3)) # height, with, color channels (R, G, B)


#Zeros and Ones
zeros = torch.zeros(size=(3,4))  
ones = torch.ones(size=(4,4)) 

#Creating a range of tensors
range_tensor = torch.arange(1, 10, 2) 

# Tensors Like 
one_to_ten = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
ten_zeros = torch.zeros_like(one_to_ten) 
print(ten_zeros)