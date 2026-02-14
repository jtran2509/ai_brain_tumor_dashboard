import numpy as np
import os
from PIL import Image # Handle different type of images

import monai # PyTorch and Monai are good combo for Xray and MRI
from monai.config import print_config
from monai.data import decollate_batch
from monai.transforms import (
    Activations, EnsureChannelFirst, AsDiscrete, Compose, LoadImage, RandFlip,
    RandRotate, RandZoom, ScaleIntensity, Resize, ToTensor
)
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Define data paths
def get_brain_loaders(BASE_DIR, batch_size=32):
    # 1. define the path/base_dir
    TRAINING_FOLDER = os.path.join(BASE_DIR, 'Training')
    TESTING_FOLDER = os.path.join(BASE_DIR, 'Testing')
    VAL_FOLDER = os.path.join(BASE_DIR, 'Validation')
    CLASSES = sorted(os.listdir(TRAINING_FOLDER))

    # Define transformer 
    train_transform = Compose([
        lambda x: np.array(x.convert("RGB")),
        EnsureChannelFirst(channel_dim=-1),
        Resize(spatial_size=(224, 224)),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob = 0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5)
    ])

    # Transform validation dataset
    val_transformation = Compose([
        lambda x: np.array(x.convert("RGB")),
        EnsureChannelFirst(channel_dim=-1), # Moved EnsureChannelFirst before Resize
        Resize(spatial_size=(224, 224)),
        ScaleIntensity() # No use Rotation or Flip, we want to see how good our image is performing under "real" condition after learning through tons of flipping and rotating images
    ])

    # Create datasets
    train_dataset = ImageFolder(TRAINING_FOLDER, transform=train_transform)
    val_dataset = ImageFolder(VAL_FOLDER, transform=val_transformation)
    test_dataset = ImageFolder(TESTING_FOLDER, transform=val_transformation)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_dataset.classes