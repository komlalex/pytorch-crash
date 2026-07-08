import torch 
from torch import nn 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split  
from pandas import DataFrame

# Set hyperparameters 
NUM_CLASSES = 5
NUM_FEATURES = 2 
RANDOM_SEED = 42 

# Device agnostic code and random seed  
torch.manual_seed(RANDOM_SEED) 
torch.cuda.manual_seed(RANDOM_SEED) 

device = "cuda"  if torch.cuda.is_available() else "cpu" 

# Create multi-class data 
X_blob, y_blob = make_blobs(n_samples=1000,  # type: ignore
                            n_features=NUM_FEATURES, 
                            centers=NUM_CLASSES, 
                            random_state=RANDOM_SEED) 

blobs = DataFrame({"x": X_blob[:, 0], "y": X_blob[:,1], "label": y_blob}) 
#print(blobs.label.value_counts()) 

plt.scatter(x=X_blob[:,0], 
            y=X_blob[:,1], 
            c=y_blob, 
            cmap="RdYlBu") 
#plt.show() 
X_blob = torch.from_numpy(X_blob).type(torch.float).to(device) 
y_blob = torch.from_numpy(y_blob).type(torch.float).to(device).unsqueeze(dim=1) 

print(X_blob.shape, y_blob.shape)