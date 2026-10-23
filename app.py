import io
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image
import torch

from src.models import MultimodalConsistencyDetector

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
# Cached Model & Detector Loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Calibrated CLIP Multimodal Engine...")
def load_detector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MultimodalConsistencyDetector(device=device)
    return detector, device


detector, device = load_detector()

# ---------------------------------------------------------------------------
# Sidebar: Threshold Slider & Benchmark Presets
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Calibration & Settings")
    st.markdown(f"**Runtime Device**: `{device}` ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    st.markdown("**Backbone Model**: `openai/clip-vit-base-patch32`")
    st.markdown("**Calibration Engine**: Zero-Shot Contrastive Entailment + Domain Mapping")

    st.markdown("---")
    st.subheader("Decision Sensitivity")
    threshold = st.slider(
        "Consistency Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.45,
        step=0.01,
        help="Pairs with Confidence ≥ Threshold are classified as Genuine / Consistent. Pairs below are Inconsistent / Misinformation."
    )
    st.caption(f"Active Decision Threshold: **{threshold * 100:.1f}%**")

    st.markdown("---")
    st.subheader("Calibrated Benchmark Presets")
    st.write("Click to load verified test pairs:")

    PRESETS = {
        "Preset 1 (Match: Diplomatic Summit)": {
            "text": "Diplomatic Leaders Sign Historic International Peace Accord at Geneva Summit",
            "img_path": "data/images/politics_conference.jpg",
            "expected": "Consistent / Genuine Context"
        },
        "Preset 2 (Mismatch: Miami Floods on Press Conference)": {
            "text": "Severe flash floods in Miami submerge major highways and downtown districts",
            "img_path": "data/images/politics_conference.jpg",
            "expected": "Inconsistent / Out-of-Context Misinformation"
        },
        "Preset 3 (Match: Earthquake Damage)": {
            "text": "Massive Magnitude 7.2 Earthquake Strikes Coastal Province Causing Widespread Structural Damage",
            "img_path": "data/images/disaster_earthquake.jpg",
            "expected": "Consistent / Genuine Context"
        },
        "Preset 4 (Match: Quantum Microchip)": {
            "text": "Tech Giant Unveils Revolutionary Next-Generation Quantum Computing Processor Architecture",
            "img_path": "data/images/tech_processor.jpg",
            "expected": "Consistent / Genuine Context"
        },
        "Preset 5 (Match: Championship Game)": {
            "text": "Underdog Franchise Clinches National Championship in Thrilling Double Overtime Victory",
            "img_path": "data/images/sports_championship.jpg",
            "expected": "Consistent / Genuine Context"
        },
        "Preset 6 (Mismatch: Earthquake on Tech Cleanroom)": {
            "text": "Massive Magnitude 7.2 Earthquake Strikes Coastal Province Causing Widespread Structural Damage",
            "img_path": "data/images/tech_processor.jpg",
            "expected": "Inconsistent / Out-of-Context Misinformation"
        },
        "Preset 7 (Match: Political Speech)": {
            "text": "Donald Trump speaking at a formal conference",
            "img_path": "data/images/test_political_portrait.png",
            "expected": "Consistent / Genuine Context"
        }
    }

    selected_preset = st.selectbox("Choose a Benchmark Sample:", ["None"] + list(PRESETS.keys()))

# ---------------------------------------------------------------------------
# Main Layout & Input Form
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">Multimodal Fake News & Consistency Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Verify text-image contextual alignment using calibrated CLIP embeddings, semantic topic classification, and hypothesis entailment.</div>', unsafe_allow_html=True)

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
        placeholder="e.g., Diplomatic Leaders Sign Historic International Peace Accord at Geneva Summit..."
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
# Calibrated Inference & Results Section
# ---------------------------------------------------------------------------

if analyze_clicked:
    if not headline_input.strip() or active_image is None:
        st.warning("Please provide both a text headline and an image for multimodal consistency analysis.")
    else:
        with st.spinner("Executing calibrated multimodal cross-verification..."):
            res = detector.analyze(
                image=active_image,
                headline=headline_input.strip(),
                threshold=threshold
            )

            # Telemetry logging to terminal
            print("=" * 70)
            print("[Streamlit / Inference] CALIBRATION TELEMETRY:")
            print(f"  * Headline Input            : '{headline_input[:80]}...'")
            print(f"  * Raw Cosine Similarity     : {res['raw_similarity']:.4f}")
            print(f"  * Confidence Score          : {res['confidence_pct']:.1f}%")
            print(f"  * Decision Threshold        : {threshold:.4f}")
            print(f"  * Detected Image Topic      : {res['image_topic']}")
            print(f"  * Detected Headline Topic   : {res['headline_topic']}")
            print(f"  * Final Verdict             : {res['verdict']} ({res['status_color'].upper()})")
            print("=" * 70)

        st.markdown("---")
        st.subheader("Verification & Diagnostic Results")

        # -------------------------------------------------------------------
        # Prominent UI Metrics Display (4 Metric Columns)
        # -------------------------------------------------------------------
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.markdown("**1. Consistency Confidence**")
            diff = res["confidence"] - threshold
            st.metric(
                label="Confidence Score",
                value=f"{res['confidence_pct']:.1f}%",
                delta=f"{diff * 100:+.1f}% vs threshold",
                delta_color="normal" if res["is_consistent"] else "inverse"
            )
            st.progress(res["confidence"])

        with m_col2:
            st.markdown("**2. Decision Threshold**")
            st.metric(
                label="Cutoff Requirement",
                value=f"{threshold * 100:.1f}%"
            )
            st.caption("Adjustable in sidebar")

        with m_col3:
            st.markdown("**3. Visual Evidence Topic**")
            st.metric(
                label="Detected Image Domain",
                value=res["image_topic"].split("/")[0].strip()
            )
            st.caption(f"Full: {res['image_topic']}")

        with m_col4:
            st.markdown("**4. Headline Claim Topic**")
            st.metric(
                label="Detected Claim Domain",
                value=res["headline_topic"].split("/")[0].strip()
            )
            st.caption(f"Full: {res['headline_topic']}")

        st.markdown("---")

        # Prominent Verdict Banner
        if res["is_consistent"]:
            st.markdown(
                f'<div class="badge-consistent">✅ {res["verdict"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="badge-inconsistent">⚠️ {res["verdict"]}</div>',
                unsafe_allow_html=True
            )

        # Detailed Diagnostic Summary Card
        st.markdown("#### Diagnostic Explanation")
        if res["is_consistent"]:
            st.success(
                f"**{res['explanation']}**\n\n"
                f"- **Direct Cosine Similarity**: `{res['raw_similarity']:.4f}`\n"
                f"- **Hypothesis Entailment Ratio**: `{res['entailment_ratio'] * 100:.1f}%`\n"
                f"- **Cross-Modal Topic Agreement**: `{res['topic_agreement'] * 100:.1f}%`"
            )
        else:
            st.error(
                f"**{res['explanation']}**\n\n"
                f"- **Direct Cosine Similarity**: `{res['raw_similarity']:.4f}`\n"
                f"- **Hypothesis Entailment Ratio**: `{res['entailment_ratio'] * 100:.1f}%`\n"
                f"- **Cross-Modal Topic Agreement**: `{res['topic_agreement'] * 100:.1f}%`"
            )
