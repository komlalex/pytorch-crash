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

# Set random seed 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
#print(circles.label.value_counts())

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
x = torch.from_numpy(x).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device).unsqueeze(dim=1)

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
 
# Let's replicate using nn.Sequential 
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5), 
    nn.Linear(in_features=5, out_features=1), 
).to(device)

# Make some predictions with the model 
with torch.inference_mode(): 
    untrained_preds = model_0(x_test) 
    #print(untrained_preds) 

"""Optimizer and loss  
For regression you might want MAE or MSE 
For classification you might want binary cross entropy or categorical cross entropy (cross entropy)

Two of the most common optimizers are SGD and Adam
"""
loss_fn = nn.BCEWithLogitsLoss() # Sigmoid activation function built in
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1) 

# Let's calculate accuracy - out of 100 examples, what percentage does our model get right
def accuracy(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item() 
    acc = (correct/len(y_true)) * 100 
    return acc  

# Train the model 
epochs = 1000

for epoch in range(epochs): 
    model_0.train() 
    y_logits = model_0(x_train)
    y_preds = torch.round(torch.sigmoid(y_logits)) 

    # Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train) # nn.BCELossWithLogits takes raw logits
    train_acc = accuracy(y_true=y_train, 
                   y_preds=y_preds)
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step()  

    with torch.inference_mode(): 
        model_0.eval() 
        test_logits = model_0(x_test)  
        test_preds = torch.round(torch.sigmoid(test_logits))   

        # Calculate test loss/ accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy(y_true=y_test, 
                       y_preds=test_preds)   
        
        # Print out what's happening
        if epoch % 10 ==0: 
            pass
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

""" Make predictions and evaluate the model 
From the metrics it looks like our model is not learning anything.
So to inspect it. let's make some predictions and make them visual

""" 
import requests  
from pathlib import Path 

# Download helper function from Learn PyTorch repo (if it's not already downloaded) 
if Path("helper_function.py").is_file(): 
    print("helper_functions.py already exists, skipping download.") 
else: 
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
         f.write(request.content) 

from helper_functions import plot_predictions, plot_decision_boundary  

# Plot decision boundary of the model 
plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1)  
plt.title("Train") 
plot_decision_boundary(model_0, x_train, y_train) 
plt.subplot(1, 2, 2) 
plt.title("Test")
plot_decision_boundary(model_0, x_test, y_test) 
plt.show()
