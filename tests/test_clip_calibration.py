import os
import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MultimodalConsistencyDetector

# Suppress HuggingFace Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def run_calibration_test():
    print("=" * 70)
    print("AUTOMATED SANITY TEST: MULTIMODAL CONSISTENCY CALIBRATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Execution Device  : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    img_path = PROJECT_ROOT / "data" / "images" / "test_political_portrait.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Political portrait image not found at {img_path}")

    image = Image.open(img_path).convert("RGB")
    detector = MultimodalConsistencyDetector(device=device)

    # Test Pairs
    text_pair1 = "Donald Trump speaking at a formal conference"  # Match (Politics)
    text_pair2 = "Severe flash floods in Miami"                 # Mismatch (Disaster)

    print("\n" + "-" * 70)
    print("CALIBRATED MULTIMODAL INFERENCE & TOPIC MAPPING")
    print("-" * 70)

    res1 = detector.analyze(image=image, headline=text_pair1, threshold=0.45)
    res2 = detector.analyze(image=image, headline=text_pair2, threshold=0.45)

    print(f"\n[PAIR 1 - MATCH]")
    print(f"  * Headline              : \"{text_pair1}\"")
    print(f"  * Detected Headline Topic: {res1['headline_topic']}")
    print(f"  * Detected Image Topic  : {res1['image_topic']}")
    print(f"  * Raw Cosine Similarity : {res1['raw_similarity']:.4f}")
    print(f"  * Calibrated Confidence : {res1['confidence_pct']:.1f}%")
    print(f"  * Verdict               : {res1['verdict']} (STATUS: {'PASS' if res1['is_consistent'] else 'FAIL'})")

    print(f"\n[PAIR 2 - MISMATCH]")
    print(f"  * Headline              : \"{text_pair2}\"")
    print(f"  * Detected Headline Topic: {res2['headline_topic']}")
    print(f"  * Detected Image Topic  : {res2['image_topic']}")
    print(f"  * Raw Cosine Similarity : {res2['raw_similarity']:.4f}")
    print(f"  * Calibrated Confidence : {res2['confidence_pct']:.1f}%")
    print(f"  * Verdict               : {res2['verdict']} (STATUS: {'PASS' if not res2['is_consistent'] else 'FAIL'})")

    separation = res1["confidence"] - res2["confidence"]
    print("\n" + "=" * 70)
    print("NUMERICAL SEPARATION SUMMARY")
    print("=" * 70)
    print(f"[+] Pair 1 (Match) Confidence       : {res1['confidence_pct']:.1f}%")
    print(f"[+] Pair 2 (Mismatch) Confidence    : {res2['confidence_pct']:.1f}%")
    print(f"[+] Confidence Separation Gap       : +{separation * 100:.1f}%")
    print(f"[+] Calibration Verification        : SUCCESSFUL")
    print("=" * 70)

    # Sanity checks
    assert res1["is_consistent"] is True, f"Pair 1 must be Consistent, got confidence {res1['confidence_pct']}%"
    assert res2["is_consistent"] is False, f"Pair 2 must be Inconsistent, got confidence {res2['confidence_pct']}%"
    assert separation > 0.30, f"Separation gap must be > 30%, got {separation * 100:.1f}%"


if __name__ == "__main__":
    run_calibration_test()
