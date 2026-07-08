import torch 
from torch import nn 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split  
from pandas import DataFrame

# Set hyperparameters 
NUM_CLASSES = 4
NUM_FEATURES = 2 
RANDOM_SEED = 42 

# Device agnostic code and random seed  
torch.manual_seed(RANDOM_SEED) 
torch.cuda.manual_seed(RANDOM_SEED) 

device = "cuda"  if torch.cuda.is_available() else "cpu" 

# Accuracy function 
def accuracy_fn(y_true, y_pred): 
    return torch.sum( (y_true == y_pred)) / len(y_true) * 100 

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
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor).to(device)  # type: ignore

x_train, x_test, y_train, y_test = train_test_split(X_blob, y_blob, 
                                                    random_state=RANDOM_SEED, 
                                                    test_size=0.2) 

# Build Model 
class BlobModel(nn.Module): 
    def __init__(self, input_features: int, output_features: int, hidden_units=256) -> None:
        """
        Initializes multi-class classification model
        Args: 
        input_features (int): Number of input features to the model 
        output_features(int): Number of output models to the model 
        hidden_units (int): Number of hidden units between layers, default 256 

        Return: 

        Examples: 
    
        """
        super().__init__() 
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units), 
            nn.ReLU(), 
            nn.Linear(in_features=hidden_units, out_features=hidden_units), 
            nn.ReLU(), 
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.linear_layer_stack(x)
    
model_0 = BlobModel(input_features=2, 
                    output_features=4, 
                    hidden_units=256).to(device) 

# Train Model 
optimizer = torch.optim.Adam(params=model_0.parameters(), 
                             lr=0.01) 
loss_fn = torch.nn.CrossEntropyLoss() 

EPOCHS = 100


for epoch in range(EPOCHS): 
    # Training
    model_0.train() 
    y_logits = model_0(x_train) 
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) 
    loss = loss_fn(y_logits, y_train) 
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step()  

    # Testing 
    model_0.eval() 
    with torch.inference_mode():  
        test_logits = model_0(x_test) 
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)  
        test_loss = loss_fn(test_logits, y_test) 
        test_acc = accuracy_fn(y_true=y_test, 
                               y_pred = test_preds) 

    if epoch % 10 == 0: 
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.5f} Test Acc: {test_acc:.2f}%")



