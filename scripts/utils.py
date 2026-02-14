#Use MONAI to pre-processing and transform
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import os
import torch


# Pipeline class for the EDA
class EDA:
    def __init__(self, training_folder, testing_folder, valid_folder, classes):
        self.training_folder = training_folder
        self.testing_folder = testing_folder
        self.valid_folder = valid_folder
        self.classes = classes

    def plot_samples(self):
        """
        Function to plot training samples
        """
        fig, ax = plt.subplots(1, 4, figsize=(10, 6))
        for idx, data in enumerate(self._get_training_samples().items()):
            cls, path = data
            img = cv2.imread(path)
            ax[idx].imshow(img)
            ax[idx].set_title(cls)
            ax[idx].axis('off')
        plt.show()

    def plot_counts(self, set_type):
        """
        Function to plot distribution counts.
        :param self: Description
        :param set_type (str): sets to plot counts for. Options: `train` & `set`
        """
        counts = self._get_counts(set_type=set_type)
        X = list(counts.keys())
        y = list(counts.values())

        # Plot counts
        sns.barplot(x=X, y=y, palette='viridis') # Changed from sns.countplot to sns.barplot
        plt.title(f'{set_type.capitalize()} Set Distribution')
        plt.xlabel('Tumor Type', fontsize=12)
        plt.ylabel('Counts', fontsize=12)
        plt.show()

    def _get_training_samples(self):
        """
        Function to retrieve a list of one sample per each class from training folder
        """
        paths = [os.path.join(self.training_folder, cls) for cls in self.classes]
        img_paths = {} 
        for path, cls in zip(paths, self.classes):
            img_name = sorted(os.listdir(path))[0]
            img_paths[cls] = os.path.join(path, img_name)
        return img_paths

    def _get_counts(self, set_type):
        """
        Function to get the count of each class in a given set

        :param `set_type`: get counts for either `train` and `test`
        return dictionary of counts
        """
        if set_type=='train':
            folder = self.training_folder
        elif set_type == 'test':
            folder = self.testing_folder
        else:
            folder = self.valid_folder

        counts = {}

        #Iterate through class
        for class_type in self.classes:
            dir = os.path.join(folder, class_type)
            counts[class_type] = len(os.listdir(dir))

        return counts
    

# Import dependencies necessary
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

# Use 1 image from the test dataset to test
dense_model.eval()
img_tensor, label = test_dataset[60] # Takes a random image and its true label from the dataset
input_tensor = img_tensor.unsqueeze(0).to(Config.device) # (1, 3, 224, 224)

# Prepare background image for display
rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

# Run Grad-CAM
cam_image = generate_gradcam(dense_model, input_tensor, rgb_img)


# Get the model's raw output
with torch.no_grad():
  output = dense_model(input_tensor)
  predicted_class_idx = torch.argmax(output, dim=1).item()
  confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class_idx].item() *100

predicted_label = test_dataset.classes[predicted_class_idx]
true_label = test_dataset.classes[label]
# Display
plt.imshow(cam_image)
plt.title(f"True Label: {true_label} | Pred: {predicted_label}({confidence:.2f}%)")
plt.axis('off')
plt.savefig("brain_tumor_gradcam", dpi = 300, bbox_inches='tight')
plt.show()
