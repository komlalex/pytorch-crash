import torch 
from torch import nn 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path

# Device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data 
start = 0 
end = 1 
step = 0.02  
weight = 0.7 
bias = 0.3 
X = torch.arange(start=start, end=end, step=step).unsqueeze(dim=1).to(device)
Y = weight * X + bias 

# Split data 
train_split = int(len(X) * 0.8)
x_train, y_train = X[:train_split], Y[:train_split] 
x_test, y_test = X[train_split:], Y[train_split:]  
print(Y.device)
def plot_predictions(x_train = x_train.cpu(), 
                     y_train = y_train.cpu(), 
                     x_test = x_test.cpu(), 
                     y_test = y_test.cpu(), 
                     predictions = None): 
    plt.figure(figsize=(10, 15)) 
    plt.scatter(x_train, y_train, c="b", s=4, label="Training") 
    plt.scatter(x_test, y_test, c="g", s=4, label="Testing")  
    if predictions is not None: 
        plt.scatter(x_test, predictions.cpu(), c="r", s=4, label="Predictions")
    plt.legend() 
    plt.xlabel("X") 
    plt.ylabel("Y") 
    plt.show() 

#plot_predictions() 
# Build Model 
class LinearRegressionModelV2(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        # Use a linear layer to create the parameters (linear transform, probing layer, dense layer, fully-connected layer)
        self.linear_layer = nn.Linear(in_features=1, 
                                out_features=1) 
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()  

# Set the model to a target device 
model_1.to(device)  

"""Training"""
loss_fn = torch.nn.L1Loss() 
optimizer  = torch.optim.SGD(params=model_1.parameters(), 
                             lr=0.01)

torch.manual_seed(42) 
epochs = 200  

for epoch in range(epochs): 
    model_1.train() 
    y_preds = model_1(x_train) 
    loss = loss_fn(y_preds, y_train) 
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()  

    # Training 
    model_1.eval() 
    with torch.inference_mode(): 
        test_preds = model_1(x_test) 
        test_loss = loss_fn(test_preds, y_test) 

    # Print out what is happening 
    if epoch % 10 == 0: 
        pass
        #print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss}") 


# Making predictions and plotting them
with torch.inference_mode(): 
    model_1.eval() 
    preds = model_1(x_test) 
    plot_predictions(predictions=(preds)) 

# Saving and Loading a Model 
MODEL_NAME = "lin_regression_v2.pth" 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, 
                 exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME   
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)

# Loading Model 
loaded_model_1 = LinearRegressionModelV2()  
loaded_model_1.to(device) 
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH, 
                                          weights_only=True)) 

print(loaded_model_1.state_dict() == model_1.state_dict()) 

