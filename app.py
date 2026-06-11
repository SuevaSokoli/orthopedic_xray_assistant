import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gradcam import analyze_xray, preprocess_image, generate_gradcam
from src.recommender import get_recommendation, get_urgency_description
from src.model import build_model
from src.dataset import IDX_TO_BODY_PART, IDX_TO_CONDITION
from src.config import CHECKPOINTS_DIR

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Orthopedic X-Ray Assistant",
    page_icon="🦴",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .urgency-high {
        background-color: #ffeaea;
        border-left: 5px solid #dc3545;
    }
    .urgency-medium {
        background-color: #fff3e0;
        border-left: 5px solid #fd7e14;
    }
    .urgency-low {
        background-color: #eafaf1;
        border-left: 5px solid #28a745;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #999;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        return None, device, "Model not found. Please train the model first."

    model = build_model(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, device, None

# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="main-header">🦴 Orthopedic X-Ray Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Deep Learning-Based Decision Support System for Orthopedic Analysis</div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Bonescan.jpg/220px-Bonescan.jpg", width=200)
    st.markdown("### About")
    st.markdown("""
    This system uses a **ResNet50** deep learning model 
    trained on the **MURA dataset** (36,000+ X-rays) 
    to assist orthopedic doctors in analyzing X-ray images.
    
    **Capabilities:**
    - 🦴 Body part detection (7 types)
    - 🔍 Abnormality detection
    - 🗺️ Grad-CAM visualization
    - 📋 Clinical recommendations
    """)
    st.markdown("---")
    st.markdown("**Developed by:** Sueva Sokoli")
    st.markdown("**Institution:** Canadian Institute of Technology")
    st.markdown("**Year:** 2026")

# ── Load model ─────────────────────────────────────────────
model, device, error = load_model()

if error:
    st.error(f"⚠️ {error}")
    st.info("Please train the model first by running the training notebook on Google Colab.")
    st.stop()
else:
    st.success(f"✅ Model loaded successfully! Running on: {str(device).upper()}")

# ── Main interface ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 📤 Upload X-Ray Image")

uploaded_file = st.file_uploader(
    "Choose an X-ray image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = os.path.join("outputs", "temp_upload.png")
    os.makedirs("outputs", exist_ok=True)

    image = Image.open(uploaded_file).convert("RGB")
    image.save(temp_path)

    # ── Layout ─────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🖼️ Original X-Ray")
        st.image(image, use_column_width=True)

    # ── Run analysis ────────────────────────────────────────
    with st.spinner("🔍 Analyzing X-ray..."):
        image_tensor, image_np = preprocess_image(temp_path)

        gradcam_output, body_part_idx, condition_idx = generate_gradcam(
            model, image_tensor, image_np,
            target="condition", device=str(device)
        )

        body_part = IDX_TO_BODY_PART[body_part_idx]
        condition = IDX_TO_CONDITION[condition_idx]
        recommendation = get_recommendation(body_part, condition)

    with col2:
        st.markdown("#### 🗺️ Grad-CAM Heatmap")
        st.image(gradcam_output, use_column_width=True)
        st.caption("Red areas indicate regions the model focused on most.")

    with col3:
        st.markdown("#### 📊 Analysis Results")

        st.markdown(f"""
        | Finding | Result |
        |---------|--------|
        | **Body Part** | {recommendation['body_part']} |
        | **Condition** | {recommendation['condition']} |
        | **Urgency** | {recommendation['icon']} {recommendation['urgency'].upper()} |
        """)

        urgency_class = f"urgency-{recommendation['urgency']}"
        st.markdown(f"""
        <div class="result-box {urgency_class}">
            <h4>{recommendation['icon']} {get_urgency_description(recommendation['urgency'])}</h4>
            <p>{recommendation['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Confidence scores ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Model Confidence")

    with torch.no_grad():
        image_tensor_dev = image_tensor.to(device)
        body_logits, cond_logits = model(image_tensor_dev)
        body_probs = torch.softmax(body_logits, dim=1)[0]
        cond_probs = torch.softmax(cond_logits, dim=1)[0]

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Condition Confidence**")
        st.progress(float(cond_probs[condition_idx]))
        st.caption(f"{condition.title()}: {float(cond_probs[condition_idx])*100:.1f}%")

    with col5:
        st.markdown("**Body Part Confidence**")
        st.progress(float(body_probs[body_part_idx]))
        st.caption(f"{body_part.replace('XR_', '').title()}: {float(body_probs[body_part_idx])*100:.1f}%")

    # ── Save results ────────────────────────────────────────
    gradcam_save_path = os.path.join("outputs", "gradcam", "latest_gradcam.png")
    os.makedirs(os.path.dirname(gradcam_save_path), exist_ok=True)
    from src.gradcam import save_gradcam
    save_gradcam(gradcam_output, gradcam_save_path)

# ── Disclaimer ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This system is intended as a decision support tool only. 
    It should not replace professional medical judgment. 
    All findings must be verified by a qualified radiologist or orthopedic specialist.
</div>
""", unsafe_allow_html=True)