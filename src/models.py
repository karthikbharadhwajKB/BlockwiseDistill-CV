""" """

import torch.nn as nn
from torchvision.models import resnet18 


# Teacher model 
def get_teacher_model(pretrained=True):
   # Load a pre-trained ResNet-18 model
   # If pretrained is True, it will load the weights trained on ImageNet
   resnet_model = resnet18(
      weights="IMAGENET1K_V1" if pretrained else None
   )    
   # Modify the first convolutional layer to accept single-channel input
   # This is useful for grayscale images or other single-channel data 
   resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
   # Replace the final fully connected layer to output 10 classes
   resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)
   return resnet_model

# Student model
class StudentModel(nn.Module):
   def __init__(self):
      super().__init__() 
      # Define the first block of the student model
      # This block consists of a convolutional layer, ReLU activation, and max pooling
      self.block1 = nn.Sequential(
         nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2, stride=2)
      )
      # Define the second block of the student model
      # This block consists of a convolutional layer, ReLU activation, and max pooling
      self.block2 = nn.Sequential(
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
      )
      # Define the third block of the student model
      # FC layers are used for classification
      # The final output layer has 10 units for 10 classes
      self.fc = nn.Linear(16*5*5, 10)  

   def forward(self, x):
      # Forward pass through the student model
      x1 = self.block1(x)
      x2 = self.block2(x1)
      # Flatten the output from the second block
      x3 = x2.view(x2.size(0), -1)  # Flatten the tensor
      out = self.fc(x3)  # Pass through the fully connected layer
      # Return the output and intermediate features
      return out, x1, x2 