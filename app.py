import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import pickle
import numpy as np
import os
from torchvision.transforms import Grayscale
import cv2 # Keep cv2 import for image processing
# Removed matplotlib.pyplot import

# Snapchat-inspired Styling
st.set_page_config(page_title="SnapCal", layout="centered", initial_sidebar_state="auto")

st.markdown('''
    <style>
    .stApp > header, .stApp > div > div:first-child {
        background-color: #FFFC00;
    }
    .stApp > div > div {
        background-color: #FBBC04;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #00FF00 !important;
        color: white !important;
        font-size: 18px !important;
        padding: 12px 28px !important;
        border-radius: 8px !important;
        border: none !important;
        cursor: pointer !important;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .stButton>button:hover {
        background-color: #00E600 !important;
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 20px;
        font-size: 900px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: 2px;
        font-italic: italic;
    }
     h2, h3, h4, h5, h6 {
        color: #1e3a8a;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stMarkdown, .stText, .stException {
        color: #000000 !important;
        font-size: 16px;
        line-height: 1.6;
    }
    .stFileUploader label, .stCameraInput label {
        font-size: 18px;
        color: #1e3a8a;
        margin-bottom: 10px;
        display: block;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"], .stCameraInput div[data-testid="stCameraInputButton"] {
        border: 2px dashed #1e3a8a;
        padding: 20px;
        border-radius: 8px;
        background-color: #ffffcc;
        text-align: center;
    }
     .stFileUploader div[data-testid="stFileUploaderDropzone"] p, .stCameraInput div[data-testid="stCameraInputButton"] p {
        color: #1e3a8a;
        font-size: 16px;
    }
    .stCaption {
        text-align: center;
        font-style: italic;
        color: #555555;
        margin-top: 5px;
    }
    hr {
        border-top: 2px solid #1e3a8a;
        margin-top: 25px;
        margin-bottom: 25px;
    }
    </style>
    ''', unsafe_allow_html=True)

if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

class MultiInputSnapCalCNN(nn.Module):
    def __init__(self):
        super(MultiInputSnapCalCNN, self).__init__()

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

        total_flattened_features = 256 + 128 + 128
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

    def forward(self, x_rgb, x_heat, x_depth):
        features_rgb = self.features_rgb(x_rgb)
        features_rgb = self.features_rgb_cont(features_rgb)

        features_heat = self.features_mono(x_heat)
        features_heat = self.features_heat_cont(features_heat)

        features_depth = self.features_mono(x_depth)
        features_depth = self.features_depth_cont(features_depth)

        features_rgb = self.gap(features_rgb)
        features_heat = self.gap(features_heat)
        features_depth = self.gap(features_depth)

        features_rgb = features_rgb.view(features_rgb.size(0), -1)
        features_heat = features_heat.view(features_heat.size(0), -1)
        features_depth = features_depth.view(features_depth.size(0), -1)

        combined_features = torch.cat((features_rgb, features_heat, features_depth), dim=1)
        output = self.classifier(combined_features)
        return output

@st.cache_resource
def load_multi_input_model(filename):
    st.info(f"Attempting to load model state dictionary from {filename}")
    try:
        model = MultiInputSnapCalCNN()
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        st.success("MultiInputSnapCalCNN model state dictionary loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {filename}")
        return None
    except KeyError:
        st.error(f"Checkpoint file {filename} does not contain 'model_state_dict'.")
        return None
    except Exception as e:
        st.error(f"Could not load model state dictionary: {e}")
        return None

model_state_dict_path = "/content/drive/MyDrive/nutrition_model_checkpoints/best_multi_input_model.pt"

model = load_multi_input_model(model_state_dict_path)

if model is None:
     st.stop()

transform_rgb = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mono_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

grayscale_transform = transforms.Grayscale(num_output_channels=1)

def generate_mono_placeholders(pil_image):
    # Removed matplotlib import from here
    import cv2 # Moved import inside function if needed, but it's already at the top of app_py_content

    img_np = np.array(pil_image.convert('L'))

    # Generate "Heat" approximation using OpenCV colormap
    # Apply a pseudocolor map (e.g., COLORMAP_HOT)
    heatmap_colored_np = cv2.applyColorMap(img_np, cv2.COLORMAP_HOT)
    # Convert the 3-channel colored numpy array back to a 1-channel grayscale PIL image
    heat_pil = Image.fromarray(heatmap_colored_np).convert('L')

    # Generate "Depth" approximation using Gradient Magnitude (already uses cv2)
    grad_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_np = 255 - gradient_magnitude
    depth_pil = Image.fromarray(depth_np).convert('L')

    return heat_pil, depth_pil


st.title('SnapCal')

st.write("Upload your meal!🍴 or take a picture to estimate its calories.")

if st.button("Camera"):
    st.session_state.show_camera = True

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

camera_image = None
if st.session_state.show_camera:
    camera_image = st.camera_input("Or take a picture!")


image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")


if image is not None and st.button('Estimate Calories'):
    try:
        image_rgb_tensor = transform_rgb(image).unsqueeze(0)

        heat_pil, depth_pil = generate_mono_placeholders(image)

        image_heat_tensor = mono_transform(heat_pil).unsqueeze(0)
        image_depth_tensor = mono_transform(depth_pil).unsqueeze(0)

        model.to('cpu')
        model.eval()

        with torch.no_grad():
            prediction = model(image_rgb_tensor, image_heat_tensor, image_depth_tensor)
            estimated_calories = prediction.item()

        st.write("---")
        st.subheader("Processed Inputs to the Model")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original RGB", use_column_width=True)
        with col2:
            st.image(heat_pil, caption="Generated Heat Placeholder", use_column_width=True)
        with col3:
            st.image(depth_pil, caption="Generated Depth Placeholder", use_column_width=True)
        st.write("---")

        st.success(f"Estimated Calories: {estimated_calories:.2f} kcal")
        st.warning("Note: Heat and Depth inputs shown above are generated from the RGB image using simple image processing techniques as placeholders. Accuracy may be limited compared to using actual multi-modal data.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")