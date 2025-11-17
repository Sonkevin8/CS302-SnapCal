import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import os
import torchvision.models as models

# ============================================
# Model Definition (ResNetRGB)
# ============================================
class ResNetRGB(nn.Module):
    def __init__(self):
        super(ResNetRGB, self).__init__()
        # Load pre-trained ResNet50
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Use all layers except the last two (average pooling and fully connected)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.pool(self.backbone(x)).flatten(1)
        return self.fc(feat)

# ============================================
# Transforms Definition
# ============================================
val_test_transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================
# Load Model Checkpoint
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = ResNetRGB().to(device)
    checkpoint_path = os.path.join("checkpoints", "best_resnet50_rgb_retrained.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval() # Set model to evaluation mode
        st.success("ResNetRGB model loaded successfully!")
    else:
        st.error(f"Error: Model checkpoint not found at {checkpoint_path}")
    return model

# Load the model at app startup
model = load_model()


# ---- Stylish Hero Section with Background ----

background_img_url = "https://images.pexels.com/photos/70497/pexels-photo-70497.jpeg?auto=compress&fit=crop&w=1350&q=80"
icon_url = "https://cdn-icons-png.flaticon.com/512/1046/1046857.png" # food plate icon

st.set_page_config(page_title="SnapCal", layout="centered", initial_sidebar_state="auto")

st.markdown(
     f"""
    <style>
    body {{
        background: linear-gradient(120deg, #FCFF6C 0%, #f3f4f9 100%);
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }}
    .stApp {{
        background-image: url('{background_img_url}');
        background-size: cover;
        background-attachment: fixed;
    }}
    .hero-card {{
        background: rgba(255,255,255,0.80);
        border-radius: 32px;
        padding: 2.5rem 2rem 2.5rem 2rem;
        margin: 2rem auto 2rem auto;
        box-shadow: 0 7px 32px 4px rgba(16,32,45,0.10);
        max-width: 600px;
        position: relative;
        backdrop-filter: blur(2.5px);
    }}
    .snapcal-title {{
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
        font-weight: 900;
        font-size: 3rem;
        color: #1e3a8a;
        margin-bottom: .25rem;
        letter-spacing: 2.5px;
        text-shadow: 0 2px 16px #fefbde, 0 1px 4px #ffe5b1;
        text-align: center;
    }}
    .app-description {{
        color: #3d4f77;
        font-size: 1.13rem;
        text-align: center;
        margin-bottom: .5rem;
        line-height: 1.7;
    }}
    .snapcal-divider {{
        margin: 2.5rem 0 2rem 0; height:2px; background:#ffdc51; border-radius:20px;
        border:none;
    }}
    .title-icon {{
        width:55px; display:block; margin: 0 auto 6px auto; filter: drop-shadow(0 2px 6px #fff9c25e);
    }}
    .stButton>button {{
        background: linear-gradient(90deg,#ffdc51,#f8ffae 90%);
        color: #1e3a8a;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        font-size: 1.17rem;
        margin-top: .5rem;
        padding: 12px 28px;
        transition: background .2s, color .2s;
        box-shadow:0 2px 8px #efefdb42;
    }}
    .stButton>button:hover {{
        background: #ffe03c;
        color: #007bff;
    }}
    .stFileUploader > div[data-testid="stFileUploaderDropzone"] {{
        border: 2px dashed #1e3a8a;
        border-radius: 12px;
        background: #fffbe7bb;
        color: #1e3a8a;
    }}
    .stFileUploader label {{
        font-size: 1.04rem;
        color: #18306e;
        letter-spacing: 0.5px;
    }}
    .stAlert {{
        border-radius: 10px;
    }}
    </style>
  """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="hero-card">
        <img src="{icon_url}" class="title-icon" alt="SnapCal Logo">
        <div class="snapcal-title">SnapCal</div>
        <div class="app-description">
            AI-powered calorie estimation from a single photo.<br>
            Upload one meal image at a time frome above, and SnapCal will instantly analyze and predict its calories.<br>
            <span style='color:#FFD700;font-weight:600;'>Eat smart, snap fast, stay healthy.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Main App Logic (Buttons, Upload, Results) ----

st.write("")
st.markdown('<hr class="snapcal-divider"/>', unsafe_allow_html=True)

if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

if st.button("Take a Photo"):
    st.session_state.show_camera = True

uploaded_file = st.file_uploader(
    "Upload your meal photo (.png, .jpg, .jpeg)...",
    type=["png", "jpg", "jpeg"],
    key="main-uploader"
)

camera_image = st.camera_input("Or capture with webcam") if st.session_state.show_camera else None

image = None
if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image is not None:
    from PIL import Image
    image = Image.open(camera_image).convert("RGB")

if image is not None:
    st.image(image, caption="Your submitted meal", use_column_width=True)
    if st.button("Estimate Calories"):
        # Preprocess the image
        input_tensor = val_test_transform_rgb(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()

        st.markdown(
            f"""
            <div style="
            background: #fff;
            border-radius: 18px;
            box-shadow: 0px 2px 18px #e1e1e1ee;
            padding: 1.2rem 2rem;
            margin: 1rem 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            ">
            <span style="
                color: #1e3a8a;
                font-size: 2.5rem;
                font-weight: 800;
                letter-spacing: 1px;
                text-shadow: 0 2px 12px #eee;
            ">
                Estimated Calories:<br> {prediction:.1f} kcal
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Initialize FAQ toggle in session state
        if 'show_faq' not in st.session_state:
            st.session_state.show_faq = False

        # Place FAQ button at the top right corner, always visible
        faq_cols = st.columns([8, 1])
        with faq_cols[1]:
            if st.button("FAQ"):
                st.session_state.show_faq = not st.session_state.show_faq
                if st.button("Close FAQ"):
                    st.session_state.show_faq = False
        # FAQ button and display moved outside the "Estimate Calories" condition
        if 'show_faq' not in st.session_state:
            st.session_state.show_faq = False

        if st.button("FAQ"):
            st.session_state.show_faq = not st.session_state.show_faq

        if st.session_state.show_faq:
            with st.expander("Frequently Asked Questions", expanded=True):
                st.markdown("**Q: What does this estimate represent?**\n\nThis is an AI-based calorie estimate (kcal) produced from a single image. It should be taken as an approximate value, not a medical or nutritional diagnosis.")
                st.markdown("**Q: How accurate is it?**\n\nAccuracy depends on image quality, portion visibility, food diversity, and how similar the meal is to what the model saw during training. Typical errors can be significant for mixed or occluded dishes.")
                st.markdown("**Q: How should I take photos for best results?**\n\nUse a single-plate view, good lighting, minimal occlusion, and a neutral background. Top-down or 45° angled shots work well.")
                st.markdown("**Q: Is my image stored or shared?**\n\nImages are processed locally in your session unless you explicitly upload them to a remote service. The app does not automatically share images.")
                st.markdown("**Q: Can I estimate multiple items at once?**\n\nFor best results, upload one meal/plate at a time. Complex multi-item plates can reduce accuracy.")
        if st.session_state.show_faq:
            with st.expander("Frequently Asked Questions", expanded=True):
                st.markdown("**Q: What does this estimate represent?**\n\nThis is an AI-based calorie estimate (kcal) produced from a single image. It should be taken as an approximate value, not a medical or nutritional diagnosis.")
                st.markdown("**Q: How accurate is it?**\n\nAccuracy depends on image quality, portion visibility, food diversity, and how similar the meal is to what the model saw during training. Typical errors can be significant for mixed or occluded dishes.")
                st.markdown("**Q: How should I take photos for best results?**\n\nUse a single-plate view, good lighting, minimal occlusion, and a neutral background. Top-down or 45° angled shots work well.")
                st.markdown("**Q: Is My image stored or shared?**\n\nImages are processed locally in your session unless you explicitly upload them to a remote service. The app does not automatically share images.")
                st.markdown("**Q: Can I estimate multiple items at once?**\n\nFor best results, upload one meal/plate at a time. Complex multi-item plates can reduce accuracy.")
                if st.button("Close FAQ"):
                    st.session_state.show_faq = False
        st.info("Nutritional estimate based on AI analysis. For best results, use clear and well-lit meal images.")

else:
    st.write(
        """<div style="text-align:center;color:#888;font-size:1.1rem;">
        No image uploaded yet.<br>Drag and drop, browse, or use the camera above.
        </div>""",
        unsafe_allow_html=True
    )

st.write("")
st.markdown('<hr class="snapcal-divider"/>', unsafe_allow_html=True)

st.caption("© 2025 SnapCal – Creativee Technologies Capstone")


