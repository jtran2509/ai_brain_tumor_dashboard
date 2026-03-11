#Use MONAI to pre-processing and transform
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch


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

#   # Define the target layer
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

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms

def generate_gradcam(model, image_tensor, original_image_path, target_layer):
    """
    Generate GradCAM overlay without OpenCV.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        original_image_path: Path to original image file
        target_layer: The layer to hook for gradients
    
    Returns:
        overlay: RGB numpy array (H, W, 3) ready for st.image()
    """
    # 1. Get heatmap from model (this part remains unchanged)
    model.eval()
    gradients = None
    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor.unsqueeze(0))
    target_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, target_class].backward()

    handle_forward.remove()
    handle_backward.remove()

    # Pool gradients and compute heatmap
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()

    # ReLU on heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()  # normalize to [0,1]

    # 2. Load original image with PIL
    original_img = Image.open(original_image_path).convert('RGB')
    original_size = original_img.size  # (width, height)

    # 3. Resize heatmap to original image size
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(original_size, Image.BILINEAR))

    # 4. Normalize and apply colormap (Matplotlib jet)
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    colormap = plt.cm.jet(heatmap_normalized)[:, :, :3]  # RGB in [0,1]
    heatmap_colored = (colormap * 255).astype(np.uint8)

    # 5. Convert original image to numpy array
    original_np = np.array(original_img)

    # 6. Overlay: alpha blending
    alpha = 0.5
    overlay = (alpha * heatmap_colored + (1 - alpha) * original_np).astype(np.uint8)

    return overlay
