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
