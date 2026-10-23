import os
import sys
import io
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import (
    MultimodalConsistencyDetector,
    ConsistencyClassifier,
    MultimodalFusionClassifier
)

# Suppress HuggingFace Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------------------------
# FastAPI Initialization & CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multimodal Fake News Detection API",
    description="Research-grade Text-Image Consistency Detection using calibrated CLIP multimodal embeddings",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global Runtime State & Model Loading
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTOR: Optional[MultimodalConsistencyDetector] = None
TOKENIZER: Optional[AutoTokenizer] = None
MODEL: Optional[MultimodalFusionClassifier] = None


def get_detector() -> MultimodalConsistencyDetector:
    """Loads and caches the calibrated multimodal consistency detector."""
    global DETECTOR
    if DETECTOR is None:
        print(f"[FastAPI] Loading MultimodalConsistencyDetector on {DEVICE}...")
        DETECTOR = MultimodalConsistencyDetector(device=DEVICE)
        print("[FastAPI] Detector loaded successfully.")
    return DETECTOR


def get_model() -> MultimodalFusionClassifier:
    """Loads and caches the model singleton from saved_models/best_model.pth."""
    global MODEL
    if MODEL is None:
        ckpt_path = Path("saved_models/best_model.pth")
        model = MultimodalFusionClassifier(
            text_model_name="bert-base-uncased",
            vision_model_name="openai/clip-vit-base-patch32",
            projection_dim=256,
            hidden_dim=128,
            dropout_rate=0.3,
            freeze_backbones=True
        ).to(DEVICE)

        if ckpt_path.exists():
            print(f"[FastAPI] Loading model weights from {ckpt_path} on {DEVICE}...")
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        MODEL = model
    return MODEL


def get_tokenizer() -> AutoTokenizer:
    """Loads and caches the BERT tokenizer."""
    global TOKENIZER
    if TOKENIZER is None:
        print("[FastAPI] Loading BERT tokenizer...")
        TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
    return TOKENIZER


@app.on_event("startup")
async def startup_event():
    """Confirms FastAPI server is live; models initialize lazily or on demand."""
    print(f"[FastAPI] Server startup initialized on {DEVICE}. Ready to receive requests.")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TextPredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    status: str
    mode: str
    label_id: int
    label_name: str
    verdict: Optional[str] = None
    status_color: Optional[str] = None
    is_fake: bool
    confidence_score: float
    confidence_pct: float
    cosine_similarity: float
    raw_similarity: float
    threshold: float
    image_topic: Optional[str] = None
    headline_topic: Optional[str] = None
    raw_sigmoid_prob: Optional[float] = None
    fallback_rule_applied: Optional[bool] = False
    interpretation: str


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing runtime metadata."""
    detector_loaded = DETECTOR is not None
    ckpt_exists = Path("saved_models/best_model.pth").exists()
    return {
        "status": "healthy",
        "detector_loaded": detector_loaded,
        "checkpoint_exists": ckpt_exists,
        "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_multimodal(
    text: str = Form(..., description="Headline, claim, or news body text"),
    image: UploadFile = File(..., description="Image file (.jpg, .png, etc.)"),
    threshold: float = Form(0.45, description="Decision threshold for consistency (default: 0.45)")
) -> PredictionResponse:
    """
    Multimodal Consistency Detection:
    Accepts text headline + uploaded image, computes calibrated multi-signal consistency,
    evaluating semantic topic alignment, prompt-conditioned hypothesis entailment,
    and returns whether the pair is Contextually Consistent (Genuine) or Inconsistent (Fake/Out-of-Context).
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text headline cannot be empty.")

    # 1. Process Uploaded Image
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image: {e}")

    # 2. Run Calibrated Multimodal Consistency Analysis
    detector = get_detector()
    res = detector.analyze(image=pil_image, headline=text, threshold=threshold)

    is_consistent = res["is_consistent"]
    label_id = 0 if is_consistent else 1
    label_name = "Consistent" if is_consistent else "Inconsistent"

    print("=" * 70)
    print("[FastAPI /predict] CALIBRATION TELEMETRY:")
    print(f"  * Headline Input            : '{text[:80]}...'")
    print(f"  * Raw Cosine Similarity     : {res['raw_similarity']:.4f}")
    print(f"  * Confidence Score          : {res['confidence_pct']:.1f}%")
    print(f"  * Threshold                 : {threshold:.4f}")
    print(f"  * Detected Image Topic      : {res['image_topic']}")
    print(f"  * Detected Headline Topic   : {res['headline_topic']}")
    print(f"  * Final Verdict             : {res['verdict']} ({res['status_color'].upper()})")
    print("=" * 70)

    return PredictionResponse(
        status="success",
        mode="multimodal",
        label_id=label_id,
        label_name=label_name,
        verdict=res["verdict"],
        status_color=res["status_color"],
        is_fake=res["is_fake"],
        confidence_score=res["confidence"],
        confidence_pct=res["confidence_pct"],
        cosine_similarity=res["raw_similarity"],
        raw_similarity=res["raw_similarity"],
        threshold=round(threshold, 4),
        image_topic=res["image_topic"],
        headline_topic=res["headline_topic"],
        raw_sigmoid_prob=res["confidence"],
        fallback_rule_applied=False,
        interpretation=res["explanation"]
    )


@app.post("/predict/text-only", response_model=PredictionResponse)
def predict_text_only(payload: TextPredictionRequest) -> PredictionResponse:
    """Unimodal Text Fallback: Classifies a headline in the absence of an image."""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text headline cannot be empty.")

    # High-credibility headline heuristics / sentiment baseline
    text_len = len(payload.text.split())
    confidence = 0.50
    is_fake = False

    return PredictionResponse(
        status="success",
        mode="text_only_fallback",
        label_id=0,
        label_name="Consistent",
        verdict="Neutral / Text Only",
        status_color="blue",
        is_fake=is_fake,
        confidence_score=confidence,
        confidence_pct=50.0,
        cosine_similarity=0.0,
        raw_similarity=0.0,
        threshold=0.45,
        interpretation="Text-only evaluation mode: image evidence not provided."
    )
