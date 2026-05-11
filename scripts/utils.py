#Use MONAI to pre-processing and transform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image


# Import dependencies necessary
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

# # 1. Define Config class
# class Config:
#     batch_size = 32
#     epochs = 50
#     lr = 1e-4
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def generate_gradcam(model, input_tensor, input_image, target_category=None):
#   """
#   Args:
#     model: our trained DenseNet121
#     input_tensor: the 4D tensor (1, 3, 224, 224)
#     input_image: the original image normalized to [0, 1] for visualization
#     target_category: the index of the class you want to explain (e.g., 0 for glioma)
#   """

  # Define the target layer
#   target_layers = [model.features[-2]] # Changed target layer to the last BatchNorm2d for DenseNet121

#   # 2. Initialize GradCAM
#   cam = GradCAM(model=model, target_layers=target_layers)

#   # 3. If target_category is None, it'll use the highest scoring category
#   targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
#   # If no category, default to the highest scoring class

#   # 4. Generate mask
#   grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
#   grayscale_cam = grayscale_cam[0, :] # Take the first image in batch

#   # 5.Overlay the heatmap on the original image
#   visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

#   return visualization

def generate_gradcam(model, input_tensor, original_image):
    """
    Generate Grad-CAM heatmap overlay without OpenCV.

    Args:
        model (torch.nn.Module): Trained PyTorch model (in eval mode).
        input_tensor (torch.Tensor): Preprocessed image tensor of shape (1, C, H, W)
                                     on the same device as the model.
        original_image (PIL.Image): Original RGB image (any size).

    Returns:
        np.ndarray: Overlay image as RGB numpy array (H, W, 3) with values in [0, 255].
    """
    model.eval()

    # ----- Automatically find the last Conv2d layer -----
    target_layer = None
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        raise ValueError("No Conv2d layer found in the model.")

    # ----- Hook setup -----
    gradients = None
    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # ----- Forward pass (gradients enabled) -----
    output = model(input_tensor)                # input_tensor already has batch dimension
    target_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, target_class].backward()

    # ----- Clean up hooks -----
    forward_handle.remove()
    backward_handle.remove()

    # ----- Compute heatmap -----
    # Global average pooling of gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # shape: (C,)
    # Weight activation channels
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    # Average across channels and apply ReLU
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8  # normalize to [0,1]

    # ----- Prepare overlay (no OpenCV) -----
    # Convert original PIL image to numpy array (RGB)
    original_np = np.array(original_image.convert("RGB"))
    orig_h, orig_w = original_np.shape[:2]

    # Resize heatmap to original image size
    heatmap_resized = np.array(Image.fromarray(heatmap).resize((orig_w, orig_h), Image.BILINEAR))

    # Normalize heatmap to [0,1] again (after resize)
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)

    # Apply jet colormap using matplotlib
    colormap = plt.cm.jet(heatmap_norm)[:, :, :3]  # RGB in [0,1]
    heatmap_colored = (colormap * 255).astype(np.uint8)

    # Alpha blend
    alpha = 0.5
    overlay = (alpha * heatmap_colored + (1 - alpha) * original_np).astype(np.uint8)

    return overlay