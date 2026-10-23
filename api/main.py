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

from src.models import MultimodalFusionClassifier

# Suppress HuggingFace Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------------------------
# FastAPI Initialization & CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multimodal Fake News Detection API",
    description="Research-grade Text-Image Consistency Detection using BERT and CLIP",
    version="1.0.0"
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
MODEL: Optional[MultimodalFusionClassifier] = None
TOKENIZER: Optional[AutoTokenizer] = None
CLIP_MODEL: Optional[CLIPModel] = None
CLIP_PROCESSOR: Optional[CLIPProcessor] = None

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_model() -> MultimodalFusionClassifier:
    """Loads and caches the model singleton from saved_models/best_model.pth."""
    global MODEL
    if MODEL is None:
        ckpt_path = Path("saved_models/best_model.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at '{ckpt_path}'. Run src/train.py first.")
        
        print(f"[FastAPI] Loading model weights from {ckpt_path} on {DEVICE}...")
        model = MultimodalFusionClassifier(
            text_model_name="bert-base-uncased",
            vision_model_name="openai/clip-vit-base-patch32",
            projection_dim=256,
            hidden_dim=128,
            dropout_rate=0.3,
            freeze_backbones=True
        ).to(DEVICE)

        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        MODEL = model
        print("[FastAPI] Model loaded successfully.")
    return MODEL


def get_tokenizer() -> AutoTokenizer:
    """Loads and caches the BERT tokenizer."""
    global TOKENIZER
    if TOKENIZER is None:
        print("[FastAPI] Loading BERT tokenizer...")
        TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
    return TOKENIZER


def get_clip_model() -> CLIPModel:
    """Loads and caches the full CLIP model for direct multimodal cross-modal similarity."""
    global CLIP_MODEL
    if CLIP_MODEL is None:
        print("[FastAPI] Loading CLIP model (openai/clip-vit-base-patch32)...")
        CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE)
        CLIP_MODEL.eval()
        print("[FastAPI] CLIP model loaded successfully.")
    return CLIP_MODEL


def get_clip_processor() -> CLIPProcessor:
    """Loads and caches the CLIP processor."""
    global CLIP_PROCESSOR
    if CLIP_PROCESSOR is None:
        print("[FastAPI] Loading CLIP processor...")
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return CLIP_PROCESSOR


@app.on_event("startup")
async def startup_event():
    """Confirms FastAPI server is live; models initialize lazily or on demand."""
    print("[FastAPI] Server startup initialized. Ready to receive requests.")


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
    raw_sigmoid_prob: Optional[float] = None
    fallback_rule_applied: Optional[bool] = False
    interpretation: str


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing runtime metadata."""
    model_loaded = MODEL is not None
    clip_loaded = CLIP_MODEL is not None
    ckpt_exists = Path("saved_models/best_model.pth").exists()
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "clip_loaded": clip_loaded,
        "checkpoint_exists": ckpt_exists,
        "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_multimodal(
    text: str = Form(..., description="Headline, claim, or news body text"),
    image: UploadFile = File(..., description="Image file (.jpg, .png, etc.)"),
    threshold: float = Form(0.18, description="Decision threshold for consistency (default: 0.18)")
) -> PredictionResponse:
    """
    Multimodal Consistency Detection:
    Accepts text headline + uploaded image, computes L2-normalized CLIP cosine similarity,
    and classifies whether the pair is Contextually Consistent (>= threshold) or Inconsistent/Fake (< threshold).
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text headline cannot be empty.")

    # 1. Process Uploaded Image
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image: {e}")

    # 2. Extract CLIP embeddings with explicit L2 normalization
    clip_model = get_clip_model()
    clip_processor = get_clip_processor()

    with torch.no_grad():
        text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
        image_inputs = clip_processor(images=pil_image, return_tensors="pt").to(DEVICE)

        text_features = clip_model.get_text_features(**text_inputs)
        image_features = clip_model.get_image_features(**image_inputs)

        # Explicit Unit L2 Normalization
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Raw dot product cosine similarity
        raw_similarity = float((text_features @ image_features.T).item())

    # 3. Dynamic Threshold Classification
    if raw_similarity >= threshold:
        label = "Consistent / Likely Genuine Context"
        is_fake = False
        label_id = 0
        label_name = "Consistent"
        status_color = "green"
    else:
        label = "Inconsistent / Out-of-Context Misinformation"
        is_fake = True
        label_id = 1
        label_name = "Inconsistent"
        status_color = "red"

    print("=" * 70)
    print("[FastAPI /predict] CALIBRATION TELEMETRY:")
    print(f"  * Headline Input            : '{text[:80]}...'")
    print(f"  * Raw Cosine Similarity     : {raw_similarity:.4f}")
    print(f"  * Decision Threshold        : {threshold:.4f}")
    print(f"  * Numerical Gap             : {raw_similarity - threshold:+.4f}")
    print(f"  * Final Verdict             : {label} ({status_color.upper()})")
    print("=" * 70)

    diff = raw_similarity - threshold
    confidence = round(min(max(abs(diff) * 4.0 + 0.50, 0.50), 0.99), 4)

    if not is_fake:
        interpretation = (
            f"The image and headline display genuine cross-modal alignment. "
            f"Raw cosine similarity is {raw_similarity:.4f}, exceeding the threshold ({threshold:.4f}) by {diff:+.4f}."
        )
    else:
        interpretation = (
            f"The image and headline display cross-modal contextual inconsistency. "
            f"Raw cosine similarity is {raw_similarity:.4f}, falling strictly below the threshold ({threshold:.4f}) by {diff:+.4f}."
        )

    return PredictionResponse(
        status="success",
        mode="multimodal",
        label_id=label_id,
        label_name=label_name,
        verdict=label,
        status_color=status_color,
        is_fake=is_fake,
        confidence_score=confidence,
        confidence_pct=round(confidence * 100, 2),
        cosine_similarity=round(raw_similarity, 4),
        raw_similarity=round(raw_similarity, 4),
        threshold=round(threshold, 4),
        interpretation=interpretation
    )


@app.post("/predict/text-only", response_model=PredictionResponse)
def predict_text_only(payload: TextPredictionRequest) -> PredictionResponse:
    """Unimodal Text Fallback: Classifies a headline in the absence of an image."""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text headline cannot be empty.")

    tokenizer = get_tokenizer()
    encoding = tokenizer(
        payload.text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    model = get_model()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, images=None)

    raw_prob = float(out["probability"].item())
    is_fake = raw_prob >= 0.5
    label_id = 1 if is_fake else 0
    label_name = "Inconsistent" if is_fake else "Consistent"
    confidence = raw_prob if is_fake else (1.0 - raw_prob)

    return PredictionResponse(
        status="success",
        mode="text_only",
        label_id=label_id,
        label_name=label_name,
        verdict="Consistent / Likely Genuine Context" if not is_fake else "Inconsistent / Out-of-Context Misinformation",
        status_color="green" if not is_fake else "red",
        is_fake=is_fake,
        confidence_score=round(confidence, 4),
        confidence_pct=round(confidence * 100, 2),
        cosine_similarity=0.0,
        raw_similarity=0.0,
        threshold=0.5,
        interpretation=f"Text-only evaluation resulted in a confidence of {confidence*100:.1f}%."
    )
