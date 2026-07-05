"""PyTorch Workflow
Let's explore PyTorch end-to-end workflow 
1. Prepare and Load Data 
2. Build model 
3. Fit model to data (training)
4. Make predictions and evalute the model (inference) 
5. Save and Load the model 
6. Putting it all together 
"""
import torch 
from torch import nn # nn contains all of   PyTorch's building blocks for neural networks
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt   

# Check PyTorch version 
#print(torch.__version__)  

"""
1. Data (Preparing and Loading)  
Data can be almost anything... in machine learning. 
* Excel spreadsheet 
* Images of any kind 
* Videos
* Audio like songs 
* DNA 
* Text 

Machine learning is a game of two parts 
1. Get data into a numerical representation 
2. Build a model to learn patterns in the numerical representation 

Let's showcase this with some "known" data using the linear regression fomula.

We'll use a linear regression formula to make a straight line with known parameters
"""
# Create "known" parametwers 
weight = 0.7 
bias = 0.3 

# Create data 
start = 0 
end = 1 
step = 0.02 

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias 

"""Splitting data into training and test sets 
This is one of the most important concepts in machine learning
""" 

# Create a train test split 
train_split = int(0.8 * len(X)) 
X_train, y_train = X[:train_split], y[:train_split] 
X_test, y_test = X[train_split:], y[train_split:] 

# Visualizing our data  
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None): 
    """Plots training data, test data and compares predictions""" 
    plt.figure(figsize=(10, 7))
    # Plot training data in blue 
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data") 
    # plot training data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data") 

    # Plot predictions if they exist
    if predictions is not None: 
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")  
    
    # Show the legend
    plt.legend(prop={"size":14})
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()
    
plot_predictions()