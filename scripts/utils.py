#Use MONAI to pre-processing and transform
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import torch


# Import dependencies necessary
import pytorch_grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# # 1. Define Config class
# class Config:
#     batch_size = 32
#     epochs = 50
#     lr = 1e-4
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_gradcam(model, input_tensor, input_image, target_category=None):
  """
  Args:
    model: our trained DenseNet121
    input_tensor: the 4D tensor (1, 3, 224, 224)
    input_image: the original image normalized to [0, 1] for visualization
    target_category: the index of the class you want to explain (e.g., 0 for glioma)
  """

  # Define the target layer
  target_layers = [model.features[-2]] # Changed target layer to the last BatchNorm2d for DenseNet121

  # 2. Initialize GradCAM
  cam = GradCAM(model=model, target_layers=target_layers)

  # 3. If target_category is None, it'll use the highest scoring category
  targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
  # If no category, default to the highest scoring class

  # 4. Generate mask
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
  grayscale_cam = grayscale_cam[0, :] # Take the first image in batch

  # 5.Overlay the heatmap on the original image
  visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

  return visualization

