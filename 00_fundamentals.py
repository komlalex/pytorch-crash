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

