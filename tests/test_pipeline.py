import os
import sys
import json
import unittest
from pathlib import Path

import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MultimodalFusionClassifier
from transformers import AutoTokenizer
from torchvision import transforms

# Suppress HuggingFace Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class TestEndToEndPipeline(unittest.TestCase):
    """
    End-to-End Pipeline & QA Verification Test Suite.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.ckpt_path = PROJECT_ROOT / "saved_models" / "best_model.pth"
        cls.results_dir = PROJECT_ROOT / "results"
        cls.cm_file = cls.results_dir / "confusion_matrix.png"
        cls.metrics_file = cls.results_dir / "metrics_summary.json"
        cls.dataset_file = PROJECT_ROOT / "data" / "dataset.csv"
        cls.images_dir = PROJECT_ROOT / "data" / "images"

    def test_01_trained_model_inference(self):
        """
        Test 1: Model Inference
        Loads saved_models/best_model.pth and runs a forward pass with dummy text
        and dummy image to verify predictions, confidence scores, and cosine
        similarities are generated without runtime errors.
        """
        print("\n[RUNNING TEST 1] Model Loading & Multimodal Forward Inference...")
        self.assertTrue(self.ckpt_path.exists(), f"Model checkpoint not found at {self.ckpt_path}")

        # Instantiate model architecture
        model = MultimodalFusionClassifier(
            text_model_name="bert-base-uncased",
            vision_model_name="openai/clip-vit-base-patch32",
            projection_dim=256,
            hidden_dim=128,
            dropout_rate=0.3,
            freeze_backbones=True
        ).to(self.device)

        # Load trained weights
        checkpoint = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.assertIn("model_state_dict", checkpoint, "Checkpoint missing 'model_state_dict'")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Prepare dummy test batch
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dummy_text = ["Severe earthquake strikes coastal province", "Tech giant reveals next-gen quantum processor"]
        encoding = tokenizer(
            dummy_text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        dummy_images = torch.randn(len(dummy_text), 3, 224, 224, device=self.device)

        # Execute Forward Pass
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, images=dummy_images)

        # Verify output keys and tensor shapes
        self.assertIn("probability", out)
        self.assertIn("cos_sim", out)
        self.assertIn("text_features", out)
        self.assertIn("image_features", out)

        probs = out["probability"]
        cos_sims = out["cos_sim"]

        self.assertEqual(probs.shape, (2, 1))
        self.assertEqual(cos_sims.shape, (2, 1))

        # Value bounds validation
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities must be in [0, 1]")
        self.assertTrue((cos_sims >= -1.01).all() and (cos_sims <= 1.01).all(), "Cosine similarity must be in [-1, 1]")

        print(f"  [+] Output Probabilities : {probs.squeeze(-1).tolist()}")
        print(f"  [+] Cosine Similarities  : {cos_sims.squeeze(-1).tolist()}")
        print("  [+] Test 1 PASSED: Model correctly produces calibrated predictions and cosine similarity.")

    def test_02_artifacts_integrity(self):
        """
        Test 2: Artifact Check
        Verifies results/confusion_matrix.png and results/metrics_summary.json
        exist and are non-empty.
        """
        print("\n[RUNNING TEST 2] Artifact Presence and Integrity Checks...")
        
        # 1. Confusion Matrix Image Check
        self.assertTrue(self.cm_file.exists(), f"Confusion matrix missing at {self.cm_file}")
        cm_size = os.path.getsize(self.cm_file)
        self.assertGreater(cm_size, 1000, f"Confusion matrix file is suspiciously small ({cm_size} bytes)")
        print(f"  [+] Confusion Matrix PNG : Verified ({cm_size:,} bytes)")

        # 2. Metrics Summary JSON Check
        self.assertTrue(self.metrics_file.exists(), f"Metrics summary missing at {self.metrics_file}")
        metrics_size = os.path.getsize(self.metrics_file)
        self.assertGreater(metrics_size, 50, f"Metrics file is too small ({metrics_size} bytes)")

        with open(self.metrics_file, "r") as f:
            metrics_data = json.load(f)

        self.assertIn("multimodal_fusion", metrics_data)
        mm_metrics = metrics_data["multimodal_fusion"]
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            self.assertIn(key, mm_metrics, f"Missing metric key: {key}")
            self.assertIsInstance(mm_metrics[key], (int, float))

        print(f"  [+] Metrics Summary JSON : Verified (Acc={mm_metrics['accuracy']*100:.2f}%, F1={mm_metrics['f1_score']*100:.2f}%)")
        print("  [+] Test 2 PASSED: All generated evaluation artifacts are present and well-formed.")

    def test_03_dataset_benchmark_integrity(self):
        """
        Test 3: Benchmark Dataset Check
        Verifies data/dataset.csv has >= 250 rows and images directory has generated PNGs.
        """
        print("\n[RUNNING TEST 3] Benchmark Dataset Integrity...")
        self.assertTrue(self.dataset_file.exists(), "Dataset CSV not found")
        
        import pandas as pd
        df = pd.read_csv(self.dataset_file)
        self.assertGreaterEqual(len(df), 250, f"Dataset has fewer than 250 samples ({len(df)})")
        
        # Check image files exist
        png_count = len(list(self.images_dir.glob("*.png")))
        self.assertGreaterEqual(png_count, 250, f"Images directory has only {png_count} images")

        print(f"  [+] Dataset CSV Rows     : {len(df)} samples (>= 250 required)")
        print(f"  [+] Synthetic Images     : {png_count} files in {self.images_dir}")
        print("  [+] Test 3 PASSED: Benchmark dataset verified.")

    def test_04_critical_mismatched_pair_sanity_rule(self):
        """
        Test 4: Critical Mismatched Pair & Cosine Fallback Rule Verification
        Checks that 'Severe flash floods submerge downtown streets in Miami' paired with
        the political portrait is correctly flagged as 'Inconsistent / Misleading'
        with cosine similarity < 0.20 and label_id = 1.
        """
        print("\n[RUNNING TEST 4] Critical Mismatched Pair & Cosine Sanity Rule...")
        import requests
        portrait_path = self.images_dir / "test_political_portrait.png"
        self.assertTrue(portrait_path.exists(), f"Test portrait image missing at {portrait_path}")

        url = "http://127.0.0.1:8000/predict"
        data = {"text": "Severe flash floods submerge downtown streets in Miami"}
        with open(portrait_path, "rb") as f:
            files = {"image": ("test_political_portrait.png", f, "image/png")}
            resp = requests.post(url, data=data, files=files)

        self.assertEqual(resp.status_code, 200, f"Expected HTTP 200, got {resp.status_code}: {resp.text}")
        result = resp.json()

        print(f"  [+] Status Code          : {resp.status_code}")
        print(f"  [+] Prediction Label     : {result['label_name']} (id={result['label_id']})")
        print(f"  [+] Is Fake Flag         : {result['is_fake']}")
        print(f"  [+] Cosine Similarity    : {result['cosine_similarity']}")
        print(f"  [+] Raw Sigmoid Prob     : {result.get('raw_sigmoid_prob')}")
        print(f"  [+] Fallback Triggered   : {result.get('fallback_rule_applied')}")

        # Strict assertions
        self.assertEqual(result["label_id"], 1, "Expected label_id=1 (Inconsistent)")
        self.assertEqual(result["label_name"], "Inconsistent", "Expected label_name='Inconsistent'")
        self.assertTrue(result["is_fake"], "Expected is_fake=True")
        self.assertLess(result["cosine_similarity"], 0.20, f"Expected cosine similarity < 0.20, got {result['cosine_similarity']}")
        print("  [+] Test 4 PASSED: Critical test pair correctly flagged as Inconsistent via Sanity Rule.")


if __name__ == "__main__":
    print("=" * 65)
    print("END-TO-END QA & DEPLOYMENT TEST SUITE")
    print("=" * 65)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEndToEndPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
