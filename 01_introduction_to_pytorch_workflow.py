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
    
#plot_predictions()  

"""BUILDING THE PYTORCH MODEL 

The model adjusts the weights and biases through two main algorithms: 
1. Gradient descent 
2. Back propagation 
"""  

# Create a linear regression model class 

class LinearRegressionModel(nn.Module): 
    def __init__(self):
        super().__init__() 
        # Initialize model parameters
        self.weight = nn.Parameter(torch.randn(1, 
                                                requires_grad=True, 
                                                dtype=torch.float)) 
        self.bias = nn.Parameter(torch.randn(1, 
                                             requires_grad=True, 
                                             dtype=torch.float)) 
        
    # Forward  method to define the computation in the the model 
    def forward(self, x:torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weight * x + self.bias # Linear regression formula
    
# Create random see 
RANDOM_SEED = 42 
torch.manual_seed(RANDOM_SEED) 

# Create an instance of the model 
model_0 = LinearRegressionModel()  

# Check parameters 
params = list(model_0.parameters()) 

# List named parameters 
named_params = model_0.state_dict() 


#Make predictions with our random model  
with torch.inference_mode(): # Context manager for making predictions
    y_preds = model_0(X_test) 
    #plot_predictions(predictions=y_preds)  

# Alternatively. However torch.inference_mode() is more preferred   
""""
with torch.no_grad():
    y_preds = model_0(X_test) 
    plot_predictions(predictions=y_preds) 
"""

""" Train model 
Note: Loss function may also be called a criterion or cost function 
Loss function: A function to measure how wrong the model's predictions are compared to the ideal model
Optimizer: takes into account the loss of a model and adjusts the model's parameters (e.g. weight & bias)
to improve the loss function 

Specifically we need: 
1. Training loop
2. Testing loop 
""" 
# Setup a loss function 
loss_fn = nn.L1Loss() 
# Setup an optimizer  
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.01) # lr is a hyperparameter that defines how big/small the optimizer changes the parameters

""" Building the training and a testing loop   
1. Loop through the data 
2. FOrward pass (this involves moving data through our model's forward function to make 
predictions on the data - also called forward propagation)
3. Calculate the loss (compare forward pass predictions to ground truth labels)
4. Optimizer zero grad 
5. Loss backward - move backwards through the network to calculate the gradients 
of each of the parameters of our model with respect to the loss (backpropagation)
6. Optimizer step - Use the optimizer to adjust our model's parameters to try
improve the loss
""" 
# An epoch is one loop through the data  
torch.manual_seed(42)
epochs = 200 

# Track different values
epoch_count = [] 
train_loss_values = [] 
test_loss_values = []

# Loop through the data 
for epoch in range(epochs): 
    # Set model to training mode 
    model_0.train() # Sets all parameters that require gradient to 'True' 

    # 1. forward pass  
    y_pred = model_0(X_train)

    # 2. Calculate the loss 
    loss = loss_fn(y_pred, y_train)  

    # 3. Optimizer zero grad 
    optimizer.zero_grad()
    # 4. Perform back propagation on the loss with respect to the parameters of the model 
    loss.backward()
    # 5. Step the optimizer (perform gradient descent)  
    optimizer.step() 


    """Testing""" 
    model_0.eval() # turns off different settings in the model not needed for evaluation (dropout, batch norm layers) 
    with torch.inference_mode(): # Turns off gradient tracking  
        # 1. Forward pass  
        test_pred = model_0(X_test)
        # Calculate the loss 
        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            epoch_count.append(epoch) 
            train_loss_values.append(loss) 
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
            print(model_0.state_dict()) 

        # Plot predictions after the last epoch
        if epoch == epochs - 1:  
            print(epoch)
            y_pred =model_0(X_test)
            #plot_predictions(predictions=y_pred)  
  

# Plot loss curves  
plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label="Train loss") 
plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).numpy()), label="Test loss")  
plt.title("Training and test loss curves") 
plt.ylabel("Loss") 
plt.xlabel("Epoch") 
plt.legend()
#plt.show() 

"""Saving a model in PyTorch  
There are three main methods for saving and loading in PyTorch
1. torch.save() 
2. torch.load()
3. torch.nn.Module.load_state_dict()
""" 
# Save the model 
from pathlib import Path 
MODEL_PATH = Path("models") 
MODEL_PATH.mkdir(parents=True, exist_ok=True) 

# Model save path 
MODEL_NAME = "01_pytorch_model.pth"  
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME 

# Save the model state dict 
torch.save(obj=model_0.state_dict(), 
           f=MODEL_SAVE_PATH) 

# Load the saved model  
# Since we saved our model's state dict we create a new instance of the model and load the saved state dict
loaded_model_0 = LinearRegressionModel()   


# Load the saved state dict of model_0 
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True)) 


# Make some predictions with our loaded model 
loaded_model_0.eval() 
with torch.inference_mode(): 
    # original model
    model_0_preds = model_0(X_test)
    # loaded model 
    loaded_model_preds = loaded_model_0(X_test)  
    # Compare model preds with original model 
    print(loaded_model_preds == model_0_preds) # All True


