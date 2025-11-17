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


# ---- Header Menu with FAQ Button (SnapCal title removed) ----
header_cols = st.columns([8, 1])
with header_cols[1]:
    if st.button("FAQ", key="faq_open"):
        st.session_state.show_faq = True

# ---- FAQ Panel with Darker Background ----
if st.session_state.get('show_faq', False):
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #bdbdbd 0%, #444 100%);
            border-radius: 18px;
            box-shadow: 0px 2px 18px #2228;
            padding: 2rem 2.5rem;
            margin: 1.5rem 0;
            color: #fff;
        ">
        """,
        unsafe_allow_html=True
    )
    with st.expander("Frequently Asked Questions", expanded=True):
        st.markdown("**Q: What does this estimate represent?**\n\nThis is an AI-based calorie estimate (kcal) produced from a single image. It should be taken as an approximate value, not a medical or nutritional diagnosis.")
        st.markdown("**Q: How accurate is it?**\n\nAccuracy depends on image quality, portion visibility, food diversity, and how similar the meal is to what the model saw during training. Typical errors can be significant for mixed or occluded dishes.")
        st.markdown("**Q: How should I take photos for best results?**\n\nUse a single-plate view, good lighting, minimal occlusion, and a neutral background. Top-down or 45° angled shots work well.")
        st.markdown("**Q: Is my image stored or shared?**\n\nImages are processed locally in your session unless you explicitly upload them to a remote service. The app does not automatically share images.")
        st.markdown("**Q: Can I estimate multiple items at once?**\n\nFor best results, upload one meal/plate at a time. Complex multi-item plates can reduce accuracy.")
        if st.button("Close FAQ", key="faq_close"):
            st.session_state.show_faq = False
    st.markdown("</div>", unsafe_allow_html=True)


