import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm.notebook import tqdm
import os
import pandas as pd
from torchvision.io import decode_image
from torchvision.transforms import Compose, Resize, ConvertImageDtype, Normalize, Grayscale
import torch.utils.data as data

# === CNN Model for Multi-Input Regression ===
class MultiInputSnapCalCNN(nn.Module):
    def __init__(self):
        super(MultiInputSnapCalCNN, self).__init__()

        # Feature extraction for RGB (3 channels)
        self.features_rgb = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 448->448
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 448->224

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 224->224
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 224->112
        )

        # Feature extraction for Heat and Depth (1 channel each)
        # They share initial layers to potentially learn similar low-level features
        self.features_mono = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), # 448->448
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 448->224

            nn.Conv2d(16, 32, kernel_size=3, padding=1),# 224->224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 224->112
        )

        # Further feature extraction after initial layers, specific to each modality
        self.features_rgb_cont = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 112->112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 112->56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 56->56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 56->28
        )

        self.features_heat_cont = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 112->112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 112->56

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 56->56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 56->28
        )

        self.features_depth_cont = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 112->112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 112->56

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 56->56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 56->28
        )


        # Classifier
        # Calculate the total number of input features after flattening
        # Each modality outputs features of shape [Batch, Channels, 28, 28]
        # After GAP and Flatten: RGB=256, Heat=128, Depth=128
        total_flattened_features = 256 + 128 + 128 # Sum of final channels from each path

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_flattened_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)               # Output: calories
        )

        # Define AdaptiveAvgPool2d as a separate layer to apply before flattening
        self.gap = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x_rgb, x_heat, x_depth):
        # Process each modality
        features_rgb = self.features_rgb(x_rgb)
        features_rgb = self.features_rgb_cont(features_rgb) # Continue processing RGB

        features_heat = self.features_mono(x_heat) # Initial processing for Heat
        features_heat = self.features_heat_cont(features_heat) # Continue processing Heat

        features_depth = self.features_mono(x_depth) # Initial processing for Depth
        features_depth = self.features_depth_cont(features_depth) # Continue processing Depth


        # Combine features before the classifier
        # Apply GAP before concatenation and flattening
        features_rgb = self.gap(features_rgb)
        features_heat = self.gap(features_heat)
        features_depth = self.gap(features_depth)

        # Flatten features
        features_rgb = features_rgb.view(features_rgb.size(0), -1)
        features_heat = features_heat.view(features_heat.size(0), -1)
        features_depth = features_depth.view(features_depth.size(0), -1)


        # Concatenate features along the channel dimension (dim 1)
        combined_features = torch.cat((features_rgb, features_heat, features_depth), dim=1)

        # Pass through the classifier
        output = self.classifier(combined_features)

        return output


# Custom Neural Network

class MultiInputSnapCalCNN(nn.Module):
    def __init__(self, return_features=False):
        super(MultiInputSnapCalCNN, self).__init__()

        self.return_features = return_features

        # Feature extraction for RGB (3 channels)
        self.features_rgb = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Shared for Heat & Depth
        self.features_mono = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.features_rgb_cont = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.features_heat_cont = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.features_depth_cont = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        total_flattened_features = 256 + 128

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_flattened_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_rgb, x_heat):
        # Extract RGB features
        f_rgb = self.features_rgb(x_rgb)
        f_rgb = self.features_rgb_cont(f_rgb)
        f_rgb = self.gap(f_rgb).view(f_rgb.size(0), -1)

        # Extract gray (heat) features
        f_heat = self.features_mono(x_heat)
        f_heat = self.features_heat_cont(f_heat)
        f_heat = self.gap(f_heat).view(f_heat.size(0), -1)

        # Concatenate RGB + Gray features
        combined = torch.cat((f_rgb, f_heat), dim=1)

        # 🧩 If used for feature extraction in hybrid model
        if self.return_features:
            return combined

        # 🔮 Otherwise, produce calorie prediction directly
        return self.classifier(combined)
