"""
Multimodal Fake News Detection - CLIP-based Feature Extraction + MLP Classifier.

Architecture:
  1. Frozen CLIP (openai/clip-vit-base-patch32) extracts L2-normalised
     text features (512-d) and image features (512-d).
  2. Cosine similarity = dot(text, image) after normalisation.
  3. Concatenation: [text_features | image_features | cosine_similarity] -> 1025-d.
  4. Trainable MLP: Linear(1025->256) -> ReLU -> Dropout(0.3) -> Linear(256->1).
  5. BCEWithLogitsLoss during training; sigmoid at inference.

Label convention:
  0 = Consistent / Matched
  1 = Inconsistent / Mismatched
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# Suppress HuggingFace Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class CLIPFeatureExtractor:
    """
    Utility wrapper around a frozen CLIPModel that extracts L2-normalised
    text and image embeddings.  NOT an nn.Module - just a callable.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = CLIPModel.from_pretrained(
            model_name, use_safetensors=True
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.processor = CLIPProcessor.from_pretrained(model_name)

    # ---- low-level: tensor in, tensor out ----------------------------------
    @torch.no_grad()
    def encode_text_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised text features from pre-tokenised inputs."""
        feats = self.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return F.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised image features from pre-processed pixel values."""
        feats = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(feats, p=2, dim=-1)

    # ---- high-level: raw text/PIL in, tensor out ---------------------------
    @torch.no_grad()
    def encode_raw_text(self, texts: list) -> torch.Tensor:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def encode_raw_image(self, images) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, p=2, dim=-1)


class ConsistencyClassifier(nn.Module):
    """
    Lightweight MLP head that operates on pre-extracted CLIP features.

    Input:  [text_features(512) | image_features(512) | cosine_sim(1)] -> 1025-d
    Output: raw logit (scalar) for BCEWithLogitsLoss.
    """

    def __init__(self, input_dim: int = 1025, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features:  (B, 512) - L2-normalised CLIP text embeddings.
            image_features: (B, 512) - L2-normalised CLIP image embeddings.

        Returns dict:
            logits:       (B, 1)
            probability:  (B, 1)
            cos_sim:      (B, 1)
        """
        cos_sim = torch.sum(text_features * image_features, dim=-1, keepdim=True)  # (B,1)
        fusion = torch.cat([text_features, image_features, cos_sim], dim=-1)        # (B,1025)
        logits = self.mlp(fusion)                                                   # (B,1)
        return {
            "logits": logits,
            "probability": torch.sigmoid(logits),
            "cos_sim": cos_sim,
        }


# ---------------------------------------------------------------------------
# Verification entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("MODEL ARCHITECTURE VERIFICATION (CLIP + MLP)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    # 1. Feature extractor (frozen CLIP)
    extractor = CLIPFeatureExtractor(device=device)

    # 2. Classifier head
    classifier = ConsistencyClassifier().to(device)
    print(f"[+] MLP trainable params: {sum(p.numel() for p in classifier.parameters()):,}")

    # Dummy data
    B = 4
    dummy_text = extractor.encode_raw_text([
        "A cat sitting on a mat",
        "Donald Trump speaking at a rally",
        "Massive earthquake shakes coastal city",
        "New quantum processor breaks speed records",
    ])
    dummy_img = torch.randn(B, 512, device=device)
    dummy_img = F.normalize(dummy_img, p=2, dim=-1)

    out = classifier(dummy_text, dummy_img)
    print(f"[+] logits.shape      : {tuple(out['logits'].shape)}  [Expected: ({B}, 1)]")
    print(f"[+] probability.shape : {tuple(out['probability'].shape)}  [Expected: ({B}, 1)]")
    print(f"[+] cos_sim.shape     : {tuple(out['cos_sim'].shape)}  [Expected: ({B}, 1)]")

    # Gradient check
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(out["logits"], torch.ones(B, 1, device=device))
    loss.backward()
    print(f"[+] Loss = {loss.item():.4f} - gradient flow OK OK")
    print("=" * 65)
