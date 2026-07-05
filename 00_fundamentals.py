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


"""TENSOR DATATYPES 
float32 is the default datatype of pytorch tensors

Datatypes have to do with the precision with which it is stored 
in memory
""" 
float_32_tensor = torch.tensor([1.0, 2.0, 3.0],
                                dtype=torch.float, # type of data
                                device="cpu", # device the tensor is on
                             requires_grad=False) # Whether to track gradients 
float_16_tensor = float_32_tensor.type(torch.float16)

"""Most common errors in PyTorch and Deep Learning
1. Tensors not the right datatype
2. Tensors not on right device
3. Tensors not the right shape
"""

""" 
Manipulation Tensors (Tensor Operations) 
* Addition
* Subtraction 
* Multiplication (element-wise)
* Division
* Matrix Multiplication
""" 
# Addition 
tensor = torch.tensor([1, 2, 3]) 
tensor_plus_ten = tensor + 10  

# Subtraction
tensor_minus_ten = tensor - 10

# multiplication 
tensor_by_ten = tensor * 20

# Built-in functions 
tensor_by_ten = torch.mul(tensor, 30)

"""
Tensor Multiplication  
1. Element-wise multiplication
2. Matrix multiplication (dot product)
"""
# Element-wise 
tensor = torch.tensor([1, 2, 3]) 
tensor_by_tensor = tensor * tensor  

""" Matrix multiplication   
1. Inner dimensions must match 
2. The resulting matrix takes the shape of the outer dimensions 

Use torch.matmul or torch.mm"""
matmul_tensor = torch.matmul(torch.rand(7, 3), torch.rand(3 ,4))  

# Shapes for matrix multiplication 
tensor_A = torch.tensor([ [1, 2], 
                         [3, 4], 
                         [5, 6]
]) 
tensor_B = torch.tensor([ [7, 10], 
                         [8, 11], 
                         [9, 12]
])  

# To fix our shape issue, we can manipulate the shape of our tensors
transposed = tensor_B.T
product = torch.mm(tensor_A, transposed)  

"""
Finding the min, max, mean, sum, etc (tensor aggregation)
"""
x = torch.arange(start=0, end=100, step=10)  
minimum = torch.min(x)  
maximum = torch.max(x) 
mean = torch.mean(x.type(torch.float32)) # Requires a datatype of float32
median = torch.median(x) 

min_index = torch.argmin(x) 
max_index = torch.argmax(x)

print(max_index)