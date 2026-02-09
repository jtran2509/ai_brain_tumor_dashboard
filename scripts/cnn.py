import torch 
import torch.nn as nn
import torch.nn.functional as F

## Define 1-layer, multi-layer 
# Torch.nn, torchvision.models
"""
For loss(CrossEntropy) and Optimizer (Adam): torch.optim
"""
# Create a Model Class
class Model(nn.Module):
    # Input layer
    def __init__(self, in_features=4, h1=8, h2=8, out_features=4):
        super().__init__() # Instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    
#Pick a manual seed for randomization
torch.manual_seed(41)

# Create an instance of model
mode_1 = Model()