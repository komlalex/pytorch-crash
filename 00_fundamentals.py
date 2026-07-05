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

# Find the positional min and max 
min_index = torch.argmin(x) 
max_index = torch.argmax(x)

"""
Reshaping, Stacking, squeezing and unsqueezing tensors 

*Reshaping - reshapes an input tensor to a defined shape
* View - Return a view of an input tensor of certain shape but kep the same memory
as the original tensor 
* Stacking - combine multiple tensors on top of each other (vstack ) or 
side by side hstack
* Squeeze - removes all `1` dimensions from a tensor
* Unsqueeze - add a `1` dimension to a target tensor 
* Permute - Return a view of input with dimensions permuted (swapped) in a certain way
"""
x = torch.arange(1., 10.)  

# Ad an extra dimension 
x_reshaped = torch.reshape(x, shape=(1,9))  

# Change the view 
#z = x.view(1, 9) # Changing z changes x because a view shares the same memory as the original

# Stack tensors on top of each other 
x_stacked = torch.stack([x, x, x, x], dim=0) 

# Squeeze  - 
x_squeezed = x_reshaped.squeeze() 

# Unsqueeze 
unsqueezed = x_squeezed.unsqueeze(dim=0)

# torch.permute - rearrange the dimensions of a target tensor 
x_original = torch.rand(size=(224, 224, 3)) # [height, width, color channels]
x_permuted = x_original.permute(2, 0, 1) 


"""INDEXING 
Selecting data from tensors. Indexing in PyTorch is similar to indexing 
in NumPy
"""  
x = torch.arange(1, 10).reshape(1,3,3) 
# Let's index on our new tensor 
indexed = x[0][2][2] 

# You can also use semi-colons to select "all" of a target dimension
# Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension
indexed = x[:,:,1] 
# Get all values of the 0th dimension but only the 1 index value of 1st and 2nd
# dimension
indexed = x[:,1, 1] 
# Get  index 0 of 0th and 1st dimension and all values of 2nd dimension
indexed = x[0, 0, :]
# Index on x to return 9 
indexed = x[0,2, 2] 
# Index on x to return 3, 6, 9 
indexed = x[:,:, 2]

"""PyTorch tensors & NumPy  
NumPy is a popular scientific Python numerical computing library. 
Because of this, PyTorch has functionality to interact with it.
 """
# Data in NumPy -> PyTorch tensor 
array = np.arange(1.0, 8.0) 
tensor = torch.from_numpy(array).type(torch.float32) # Default NumPy dtype is float64 unless specified otherwise

# Change the value of array, what happens to the tensor 
array = array + 1 # Tensor is not affected

# PyTorch Tensor  -> NumPy array 
tensor = torch.ones(7) 
array = tensor.numpy()


""" REPRODUCIBILITY (Taking the random out of random) 
To reduce the randomness in neural networs and PyTorch comes the concept of 
random seed. Essentially what the random seed does is 'flavour' the randomness
"""
random_A = torch.rand(3, 4) 
random_B = torch.rand(3,4) 
#print(random_A == random_B) # All False 

# Set the random seed 
RANDOM_SEED = 42 
torch.manual_seed(RANDOM_SEED) 
random_C = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED) 
random_D = torch.rand(3, 4) 
#print(random_C == random_D) # All True

""" Running tensors and PyTorch objects on the GPUs (and making faster computations)
GPUs = faster computation 
1. Easiest - Use Google Colab 
2. Use a locally setup GPU 
3. Use cloud computing - GCP, AWS, Azure, these services allow you to rent computers 
on the cloud and access them 
""" 

# Check for GPU access with PyTorch 
#print(torch.cuda.is_available()) 

"""Setup device agnostic code   
Since PyTorch is capable of running on both the GPU and CPU, it's best practice to 
setup device agnostic code
"""
device = "cuda" if torch.cuda.is_available() else "cpu" 

# Count number of devices 
count = torch.cuda.device_count() 

# Get name of device 
device_name = torch.cuda.get_device_name() 


""" PUTTING TENSORS (AND MODELS) ON THE GPU
This is done because GPUs are faster in computations
""" 
tensor = torch.tensor([1,2,3])  # Default on cpu 
#print(tensor.device) # cpu  
# Move tensor to GPU if available  

tensor_on_gpu = tensor.to(device) 
#print(tensor_on_gpu.device) # cuda 

""" Moving tensors back to GPU
If tensor is on GPU, you can't transform it to NumPy
""" 
# tensor_on_gpu.numpy() # Error
tensor_back_on_cpu = tensor_on_gpu.cpu() 
tensor_back_on_cpu.numpy() # Works now 
print(torch.cuda.get_device_capability())