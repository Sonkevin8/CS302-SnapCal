import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models

# ========== Model Definition ==========

class ResNetRGB(nn.Module):
    """ResNet50-based regressor for calorie estimation (RGB images)."""
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.pool(self.backbone(x)).flatten(1)
        return self.fc(feat)
    
val_test_transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Load model from checkpoint (if exists)."""
    model = ResNetRGB().to(device)
    checkpoint_path = os.path.join("checkpoints", "best_resnet50_rgb_retrained.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        st.success("Model loaded successfully.")
    else:
        st.error(f"Checkpoint not found at {checkpoint_path}")
    return model

model = load_model()

# ========== FAQ & Header ==========

if "show_faq" not in st.session_state:
    st.session_state.show_faq = False

header_cols = st.columns([8, 1])
with header_cols[1]:
    if st.button("FAQ", key="faq_open"):
        st.session_state.show_faq = True

if st.session_state.show_faq:
    with st.expander("Frequently Asked Questions", expanded=True):
        st.markdown("""
            **Q: What does this estimate represent?**  
            AI-based calorie estimate (kcal) from an image. It's an approximation, not a clinical result.

            **Q: How accurate is it?**  
            Depends on image quality, portion visibility, and food diversity. Best for clearly visible, single meals!

            **Q: Is my image stored or shared?**  
            Never. The image is processed *in your session only*; not logged or sent to any third-party.

            **Q: Best photo tips?**  
            Single plate, top-down or 45°, bright/neutral background, avoid occlusion!

            **Q: Multi-item images?**  
            For accuracy, one meal at a time. Complex plates lower accuracy.
        """)
        if st.button("Close FAQ", key="faq_close"):
            st.session_state.show_faq = False

# ========== Hero Banner & Brand ==========

background_img_url = "https://images.pexels.com/photos/70497/pexels-photo-70497.jpeg?auto=compress&fit=crop&w=1350&q=80"
icon_url = "https://cdn-icons-png.flaticon.com/512/1046/1046857.png"

st.set_page_config(page_title="SnapCal", layout="centered")

st.markdown(f"""
    <style>
    .stApp {{background-image: url('{background_img_url}'); background-size: cover; background-attachment: fixed;}}
    .hero-card {{
        background: rgba(255,255,255,0.88); border-radius: 32px; padding:2.5rem 2rem; margin:2rem auto;
        box-shadow: 0 7px 32px 4px rgba(16,32,45,0.10); max-width: 600px;
        backdrop-filter: blur(3px);
    }}
    .snapcal-title {{
        font-family: 'Montserrat',sans-serif; font-weight:900; font-size: 3rem; color: #1e3a8a;
        letter-spacing:2.5px; text-shadow:0 2px 16px #fefbde,0 1.5px 5px #ffe5b1; text-align: center;
    }}
    .app-description {{color:#3d4f77; font-size:1.15rem; text-align:center; margin-bottom:.5rem; line-height:1.7;}}
    .title-icon {{width:55px;display:block;margin:0 auto 6px auto;filter:drop-shadow(0 2px 6px #fff9c25e);}}
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="hero-card">
        <img src="{icon_url}" class="title-icon" alt="SnapCal logo">
        <div class="snapcal-title">SnapCal</div>
        <div class="app-description">
            AI-powered calorie estimation from a single photo.<br>
            <span style='color:#FFD700;font-weight:600;'>Eat smart, snap fast, stay healthy.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ========== Main App & Accessible Upload ==========

st.markdown('<hr style="margin:2.5rem 0; height:2px; background:#ffdc51; border-radius:20px; border:none;"/>', unsafe_allow_html=True)

if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

col1, col2 = st.columns([2,1])
with col2:
    if st.button("Take a Photo"):
        st.session_state.show_camera = True

uploaded_file = st.file_uploader("Upload your meal photo (.png, .jpg, .jpeg)...",
    type=["png", "jpg", "jpeg"], key="main-uploader", help="Image is processed in-memory and never stored."
)

camera_image = st.camera_input("Or capture with webcam") if st.session_state.show_camera else None

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image:
    image = Image.open(camera_image).convert("RGB")

if image:
    st.image(image, caption="Your submitted meal", use_column_width=True)
    if st.button("Estimate Calories"):
        with st.spinner("Calculating..."):
            try:
                input_tensor = val_test_transform_rgb(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = model(input_tensor).item()
                st.markdown(f"""
                    <div style="background:#fff; border-radius:18px; box-shadow:0px 2px 18px #e1e1e1ee; padding:1.2rem 2rem; margin:1rem 0; display:flex; flex-direction:column; align-items:center;">
                        <span style="color:#1e3a8a; font-size:2.5rem; font-weight:800; letter-spacing:1px; text-shadow:0 2px 12px #eee;">
                            Estimated Calories:<br> {prediction:.1f} kcal
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        st.info("For best results, use clear/textured images and a single plate or bowl.")

else:
    st.write("""<div style="text-align:center;color:#888;font-size:1.14rem;">
        <strong>No image uploaded yet.</strong><br>Drag and drop, browse, or use the camera above.
        </div>""", unsafe_allow_html=True)

st.caption("© 2025 SnapCal – Creative Technologies Capstone")
