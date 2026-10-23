import io
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multimodal Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-consistent {
        background-color: #DCFCE7;
        color: #166534;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 12px 18px;
        border-radius: 10px;
        display: block;
        text-align: center;
        border: 2px solid #86EFAC;
    }
    .badge-inconsistent {
        background-color: #FEE2E2;
        color: #991B1B;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 12px 18px;
        border-radius: 10px;
        display: block;
        text-align: center;
        border: 2px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Model & Processor Loader (CLIP ViT-B/32)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CLIP neural backbone & processor...")
def load_runtime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    model.eval()
    return model, processor, device


clip_model, clip_processor, device = load_runtime()

# ---------------------------------------------------------------------------
# Sidebar: Threshold Slider & Benchmark Presets
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Calibration & Settings")
    st.markdown(f"**Runtime Device**: `{device}` ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    st.markdown("**Backbone Model**: `openai/clip-vit-base-patch32`")
    st.markdown("**Normalization**: Explicit Unit L2 Normalization")

    st.markdown("---")
    st.subheader("Interactive Decision Slider")
    # Interactive Threshold Slider required
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.35,
        value=0.18,
        step=0.01,
        help="Pairs with Cosine Similarity ≥ Threshold are classified as Consistent. Pairs below are Inconsistent."
    )
    st.caption(f"Active Threshold: **{threshold:.4f}**")

    st.markdown("---")
    st.subheader("Benchmark Test Presets")
    st.write("Click to load calibrated benchmark pairs:")

    PRESETS = {
        "Pair 1 (Match: Trump Conference)": {
            "text": "Donald Trump speaking at a formal conference",
            "img_path": "data/images/test_political_portrait.png",
            "expected": "Consistent / Likely Genuine Context"
        },
        "Pair 2 (Mismatch: Miami Floods)": {
            "text": "Severe flash floods in Miami",
            "img_path": "data/images/test_political_portrait.png",
            "expected": "Inconsistent / Out-of-Context Misinformation"
        },
        "Preset 3 (Disaster Earthquake)": {
            "text": "Massive Magnitude 7.2 Earthquake Strikes Coastal Province Causing Widespread Structural Damage",
            "img_path": "data/images/img_0001.png",
            "expected": "Benchmark Pair"
        },
        "Preset 4 (Tech Quantum AI)": {
            "text": "Tech Giant Unveils Revolutionary Next-Generation Quantum Computing Processor Architecture",
            "img_path": "data/images/img_0241.png",
            "expected": "Benchmark Pair"
        }
    }

    selected_preset = st.selectbox("Choose a Benchmark Sample:", ["None"] + list(PRESETS.keys()))

# ---------------------------------------------------------------------------
# Main Layout & Input Form
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">Multimodal Fake News & Consistency Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Verify text-image contextual alignment using calibrated CLIP embeddings and explicit L2 normalization.</div>', unsafe_allow_html=True)

# Preset state handler
default_text = ""
preset_img_bytes = None
if selected_preset != "None":
    p = PRESETS[selected_preset]
    default_text = p["text"]
    if Path(p["img_path"]).exists():
        with open(p["img_path"], "rb") as f:
            preset_img_bytes = f.read()

col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.subheader("1. Headline or Claim Text")
    headline_input = st.text_area(
        "Enter news headline, social post, or claim:",
        value=default_text,
        height=130,
        placeholder="e.g., Donald Trump speaking at a formal conference..."
    )

    st.subheader("2. Paired Visual Media")
    uploaded_file = st.file_uploader(
        "Upload associated news image (.png, .jpg, .jpeg):",
        type=["png", "jpg", "jpeg"]
    )

    # Determine active image (uploaded or preset)
    active_image = None
    if uploaded_file is not None:
        try:
            active_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
    elif preset_img_bytes is not None:
        active_image = Image.open(io.BytesIO(preset_img_bytes)).convert("RGB")

    analyze_clicked = st.button("🔍 Analyze Consistency", type="primary", use_container_width=True)

with col2:
    st.subheader("Visual Evidence Preview")
    if active_image is not None:
        st.image(active_image, caption="Loaded Evidence Image", use_container_width=True)
    else:
        st.info("Please upload an image or choose a benchmark preset to inspect.")

# ---------------------------------------------------------------------------
# Robust Calibration Inference & Results Section
# ---------------------------------------------------------------------------

if analyze_clicked:
    if not headline_input.strip() or active_image is None:
        st.warning("Please provide both a text headline and an image for multimodal consistency analysis.")
    else:
        with st.spinner("Extracting CLIP representations with explicit L2 normalization..."):
            with torch.no_grad():
                # 1. Proper Embedding Extraction with CLIPProcessor
                text_inputs = clip_processor(text=[headline_input], return_tensors="pt", padding=True).to(device)
                image_inputs = clip_processor(images=active_image, return_tensors="pt").to(device)

                text_features = clip_model.get_text_features(**text_inputs)
                image_features = clip_model.get_image_features(**image_inputs)

                # 2. Explicit L2 Normalization on both vectors
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                # 3. Raw Cosine Similarity via dot product
                raw_similarity = float((text_features @ image_features.T).item())

            # 4. Dynamic Labeling based on Threshold Slider
            if raw_similarity >= threshold:
                label = "Consistent / Likely Genuine Context"
                status_color = "green"
                is_consistent = True
            else:
                label = "Inconsistent / Out-of-Context Misinformation"
                status_color = "red"
                is_consistent = False

            # Telemetry logging to terminal
            print("=" * 70)
            print("[Streamlit / Inference] CALIBRATION TELEMETRY:")
            print(f"  * Headline Input            : '{headline_input[:80]}...'")
            print(f"  * Raw Cosine Similarity     : {raw_similarity:.4f}")
            print(f"  * Current Threshold         : {threshold:.4f}")
            print(f"  * Numerical Gap             : {raw_similarity - threshold:+.4f}")
            print(f"  * Final Verdict             : {label} ({status_color.upper()})")
            print("=" * 70)

        st.markdown("---")
        st.subheader("Inference & Verification Results")

        # -------------------------------------------------------------------
        # Prominent UI Metrics Display (3 Metric Cards)
        # -------------------------------------------------------------------
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            st.markdown("**1. Raw Cosine Similarity**")
            st.metric(
                label="Raw Dot Product Similarity",
                value=f"{raw_similarity:.4f}"
            )
            sim_pct = min(max((raw_similarity + 1.0) / 2.0, 0.0), 1.0)
            st.progress(sim_pct)

        with m_col2:
            st.markdown("**2. Current Decision Threshold**")
            diff = raw_similarity - threshold
            st.metric(
                label="Threshold (Adjustable)",
                value=f"{threshold:.4f}",
                delta=f"{diff:+.4f} vs threshold",
                delta_color="normal" if is_consistent else "inverse"
            )
            st.caption(f"Cutoff value set via sidebar slider")

        with m_col3:
            st.markdown("**3. Final Verdict**")
            if is_consistent:
                st.markdown(
                    f'<div class="badge-consistent">✅ {label}</div>',
                    unsafe_allow_html=True
                )
                st.caption("Similarity exceeds decision threshold.")
            else:
                st.markdown(
                    f'<div class="badge-inconsistent">⚠️ {label}</div>',
                    unsafe_allow_html=True
                )
                st.caption("Similarity strictly below decision threshold.")

        # Detailed Diagnostic Summary Card
        st.markdown("#### Diagnostic Summary")
        if is_consistent:
            st.success(
                f"**Verified Genuine Context**: The computed raw cosine similarity is `{raw_similarity:.4f}`, "
                f"which exceeds the decision threshold of `{threshold:.4f}` by `+{diff:.4f}`. "
                f"The visual features corroborate the headline narrative."
            )
        else:
            st.error(
                f"**Flagged as Misinformation**: The computed raw cosine similarity is `{raw_similarity:.4f}`, "
                f"which falls strictly below the decision threshold of `{threshold:.4f}` by `{diff:.4f}`. "
                f"The visual evidence does not correspond to the subject of the claim."
            )
