import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models

# ========== Gold Button Styling for All Buttons ==========
st.markdown("""
<style>
.stButton > button {
    background: linear-gradient(90deg, #FFD700 0%, #FFB300 100%) !important;
    color: #1e3a8a !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 1.13rem !important;
    box-shadow: 0 2px 12px #ffe98a55 !important;
    padding: 11px 28px !important;
    margin-top: 6px !important;
    margin-bottom: 8px !important;
    cursor: pointer !important;
    transition: background .2s,color .2s,box-shadow .18s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #FFB300 0%, #FFD700 100%) !important;
    color: #0f235e !important;
}
</style>
""", unsafe_allow_html=True)

# ========== Model Definition & Loading ==========
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


# ========== HEADER: FAQ and HOME Left-Aligned, Both Gold ==========
if "show_faq" not in st.session_state:
    st.session_state.show_faq = False
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = False

header_col1, header_col2 = st.columns([3, 7])
with header_col1:
    btn_col1, btn_col2 = st.columns(2)
    # FAQ gold button
    faq_clicked = btn_col1.button("FAQ", key="faq_toggle", help="Show or hide FAQ")
    # Home gold button
    home_clicked = btn_col2.button("Home", key="home_btn", help="Restart to Home Page")

    # Event handlers
    if faq_clicked:
        st.session_state.show_faq = not st.session_state.show_faq

if home_clicked:
    st.markdown(
        """
        <meta http-equiv="refresh" content="0">
        <script>
            window.location.reload(true);
        </script>
        """,
        unsafe_allow_html=True
    )
with header_col2:
    pass

# ========== FAQ PANEL ==========
if st.session_state.show_faq:
    with st.expander("Frequently Asked Questions", expanded=True):
        st.markdown("""
        <style>
        .faq-gradient-bg {
            background: linear-gradient(120deg, #e4e8ec 0%, #c1c7cf 100%);
            border-radius: 18px;
            padding: 20px 18px 10px 18px;
            margin: 0 -10px 0 -10px;
        }
        .faq-q {
            font-size: 1.16rem !important;
            font-weight: 900 !important;
            color: #18306e !important;
            margin-bottom: 2px;
            margin-top: 13px;
            line-height: 1.3;
            letter-spacing: 0.2px;
        }
        .faq-a {
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            color: #222 !important;
            margin-bottom: 7px;
            margin-left: 2px;
            line-height: 1.45;
        }
        </style>
        <div class="faq-gradient-bg">
            <div class="faq-q">Q: What does this estimate represent?</div>
            <div class="faq-a">AI-based calorie estimate (kcal) from an image. It's an approximation, not a clinical result.</div>

            <div class="faq-q">Q: How accurate is it?</div>
            <div class="faq-a">Depends on image quality, portion visibility, and food diversity. Best for clearly visible, single meals!</div>

            <div class="faq-q">Q: Is my image stored or shared?</div>
            <div class="faq-a">Never. The image is processed <em>in your session only</em>; not logged or sent to any third-party.</div>

            <div class="faq-q">Q: Best photo tips?</div>
            <div class="faq-a">Single plate, top-down or 45°, bright/neutral background, avoid occlusion!</div>

            <div class="faq-q">Q: Multi-item images?</div>
            <div class="faq-a">For accuracy, one meal at a time. Complex plates lower accuracy.</div>
        </div>
        """, unsafe_allow_html=True)

# ========== HERO SECTION ==========
background_img_url = "https://images.pexels.com/photos/70497/pexels-photo-70497.jpeg?auto=compress&fit=crop&w=1350&q=80"
icon_url = "https://cdn-icons-png.flaticon.com/512/1046/1046857.png"

st.set_page_config(page_title="SnapCal", layout="centered")

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url('{background_img_url}');
        background-size: cover;
        background-attachment: fixed;
    }}
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

# ========== MAIN UPLOAD SECTION ==========
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

# High-contrast mobile user tip
if st.session_state.show_camera:
    st.markdown(
        """
        <div style="
            background: #f5fbee;
            border-left: 5px solid #ffe03c;
            border-radius: 12px;
            padding: 13px;
            margin-bottom: 8px;
            color: #252525;
            font-size: 1.07rem;
            font-weight: 600;
            text-shadow: 0 1px 6px #fffbe2, 0 0px 1px #eee;
            ">
            <span style="color:#18306e; font-weight: 700;">Tip:</span><br>
            If the front (selfie) camera opens by default,
            <span style="color:#b58f00; font-weight:700;">tap the camera icon/button</span>
            to switch to the
            <span style="color:#1e3a8a; font-weight:700;">back camera</span> for best food photos.
        </div>
        """, unsafe_allow_html=True
    )
    camera_image = st.camera_input("Or capture with webcam")
else:
    camera_image = None

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
                # === Estimated Calories in white card with big font ===
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