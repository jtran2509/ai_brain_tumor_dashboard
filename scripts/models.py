import torch
from monai.networks.nets import DenseNet121

def get_model(num_classes, device):
    model = DenseNet121(spatial_dims=2, 
                        in_channels=3, 
                        out_channels=num_classes).to(device)
    
    return model