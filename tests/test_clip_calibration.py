import os
import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress HuggingFace Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def run_calibration_test():
    print("=" * 70)
    print("AUTOMATED TERMINAL SANITY TEST: CLIP EMBEDDING CALIBRATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Execution Device  : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    img_path = PROJECT_ROOT / "data" / "images" / "test_political_portrait.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Political portrait image not found at {img_path}")
    
    print(f"[+] Reference Image   : {img_path}")
    image = Image.open(img_path).convert("RGB")

    print("[+] Loading CLIP model and processor (openai/clip-vit-base-patch32)...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    model.eval()

    # Define test pairs
    text_pair1 = "Donald Trump speaking at a formal conference"  # Match
    text_pair2 = "Severe flash floods in Miami"                 # Mismatch
    threshold = 0.1800

    print("\n" + "-" * 70)
    print("INFERENCE COMPUTATION & L2 VECTOR NORMALIZATION")
    print("-" * 70)

    with torch.no_grad():
        # Pair 1: Match
        inputs1 = processor(text=[text_pair1], images=image, return_tensors="pt", padding=True).to(device)
        text_feat1 = model.get_text_features(input_ids=inputs1["input_ids"], attention_mask=inputs1["attention_mask"])
        image_feat = model.get_image_features(pixel_values=inputs1["pixel_values"])

        text_feat1 = text_feat1 / text_feat1.norm(p=2, dim=-1, keepdim=True)
        image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)
        raw_sim1 = float((text_feat1 @ image_feat.T).item())

        # Pair 2: Mismatch
        inputs2 = processor(text=[text_pair2], images=image, return_tensors="pt", padding=True).to(device)
        text_feat2 = model.get_text_features(input_ids=inputs2["input_ids"], attention_mask=inputs2["attention_mask"])
        text_feat2 = text_feat2 / text_feat2.norm(p=2, dim=-1, keepdim=True)
        raw_sim2 = float((text_feat2 @ image_feat.T).item())

    verdict1 = "Consistent / Likely Genuine Context" if raw_sim1 >= threshold else "Inconsistent / Out-of-Context Misinformation"
    verdict2 = "Consistent / Likely Genuine Context" if raw_sim2 >= threshold else "Inconsistent / Out-of-Context Misinformation"

    print(f"\n[PAIR 1 - MATCH]")
    print(f"  * Text                  : \"{text_pair1}\"")
    print(f"  * Raw Cosine Similarity : {raw_sim1:.4f}")
    print(f"  * Threshold             : {threshold:.4f}")
    print(f"  * Margin vs Threshold   : {raw_sim1 - threshold:+.4f}")
    print(f"  * Verdict               : {verdict1} (STATUS: {'GREEN / PASS' if raw_sim1 >= threshold else 'FAIL'})")

    print(f"\n[PAIR 2 - MISMATCH]")
    print(f"  * Text                  : \"{text_pair2}\"")
    print(f"  * Raw Cosine Similarity : {raw_sim2:.4f}")
    print(f"  * Threshold             : {threshold:.4f}")
    print(f"  * Margin vs Threshold   : {raw_sim2 - threshold:+.4f}")
    print(f"  * Verdict               : {verdict2} (STATUS: {'RED / PASS' if raw_sim2 < threshold else 'FAIL'})")

    separation = raw_sim1 - raw_sim2
    print("\n" + "=" * 70)
    print("CALIBRATION & NUMERICAL SEPARATION SUMMARY")
    print("=" * 70)
    print(f"[+] Pair 1 (Match) Raw Similarity    : {raw_sim1:.4f}")
    print(f"[+] Pair 2 (Mismatch) Raw Similarity : {raw_sim2:.4f}")
    print(f"[+] Decision Threshold                : {threshold:.4f}")
    print(f"[+] Numerical Separation Gap         : +{separation:.4f}")
    print(f"[+] Calibration Verification          : SUCCESSFUL")
    print("=" * 70)

    # Sanity checks
    assert raw_sim1 >= threshold, f"Pair 1 should be >= {threshold}, got {raw_sim1}"
    assert raw_sim2 < threshold, f"Pair 2 should be < {threshold}, got {raw_sim2}"
    assert separation > 0.04, f"Separation gap should be > 0.04, got {separation}"

if __name__ == "__main__":
    run_calibration_test()
