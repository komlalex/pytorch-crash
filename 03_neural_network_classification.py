"""Neural Network Classification 
Classification is the problem of predicting whetehr something is one thing 
or another """
import torch 
from torch import nn 
import numpy as np  
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt   
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Set device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Make 1000 samples 
n_samples = 1000 

# Create circles 
x, y = make_circles(n_samples, 
                    noise=0.03, 
                    random_state=42)
#print(f"First 5 samples of x:\n {x[:5]}") 
#print(f"First 5 samples of y:\n {y[:5]}")  

# Make DataFrame of circles data 
circles = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "label": y}) 

# Visualize 
plt.scatter(x=x[:,0], 
            y=x[:,1], 
            c=y, 
            cmap= mpl.colormaps["RdYlBu"]
            )  
#plt.show() 

"""The data we are working with is is a toy dataset
because it is small enough to experiment but still sizeable enough to practice the 
fundamentals
""" 
# Turn data into tensors  
x = torch.from_numpy(x).type(torch.float).to(device).unsqueeze(dim=1)
y = torch.from_numpy(y).type(torch.float).to(device)

# Split into training and test sets 
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Build Model 

class CircleModelV1(nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 
        self.layer_1 = nn.Linear(in_features=2, 
                                        out_features=5) 
        self.layer_2 = nn.Linear(in_features=5, 
                                        out_features=1) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x)) 
    
# instantiate our circle model
model_0 = CircleModelV1().to(device) 
