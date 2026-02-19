import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121


def get_model(num_classes, device):
    model = DenseNet121(spatial_dims=2, 
                        in_channels=3, 
                        out_channels=num_classes).to(device)
    
    return model

    
class CNN6layer(nn.Module):
  def __init__(self, num_classes= 4):
    super(CNN6layer, self).__init__()
    # 6 layers: 4 Convolutional layers and 2 fully-connected layers
    # Layer 1: Convolutional layer ()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(16)
    # Layer 2
    self.conv2 = nn.Conv2d(in_channels=16, out_channels =32, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    # Layer 3:
    self.conv3 = nn.Conv2d(in_channels=32, out_channels =64, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(64)
    # Layer 4:
    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn4 = nn.BatchNorm2d(128)
    # Layer 5:
    self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.bn5 = nn.BatchNorm2d(256)
    # Layer 6:
    self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    self.bn6 = nn.BatchNorm2d(512)

    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(512*3*3, 512)
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512, num_classes)

  def forward(self, x):
    # Conv -> BN -> ReLU -> Pool
    x = self.pool(F.relu((self.bn1(self.conv1(x)))))
    x = self.pool(F.relu((self.bn2(self.conv2(x)))))
    x = self.pool(F.relu((self.bn3(self.conv3(x)))))
    x = self.pool(F.relu((self.bn4(self.conv4(x)))))
    x = self.pool(F.relu((self.bn5(self.conv5(x)))))
    x = self.pool(F.relu((self.bn6(self.conv6(x)))))
    # Flatten
    x= x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x