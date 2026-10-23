# 🔍 Multimodal Fake News Detection

A multimodal deep learning system designed to detect inconsistent or misleading news stories by analyzing cross-modal relationships between article claims and accompanying images.

---

## 📌 Architecture & Calibrated Verification Engine

The system uses a multimodal architecture powered by **CLIP** (`openai/clip-vit-base-patch32`) combined with a **Calibrated Zero-Shot Consistency Engine**:

1. **Semantic Topic Mapping**:
   - Classifies visual and textual representations across core news domains (Politics, Disasters, Technology, Sports, Business, etc.).
   - Detects domain mismatches (e.g. an emergency claim paired with a political press conference image).
2. **Contrastive Hypothesis Entailment**:
   - Evaluates the claim against counterfactual baselines to eliminate modality gap and text-length biases inherent in raw cosine similarity.
   - Computes an entailment ratio: $P(\text{Claim} \mid \text{Image})$.
3. **Calibrated Confidence Score**:
   - Synthesizes direct cosine similarity, entailment ratio, and semantic domain agreement into a calibrated confidence score ($0\% - 100\%$).
   - Pairs with confidence $\ge 45\%$ are classified as **Consistent / Genuine Context**.
   - Pairs with confidence $< 45\%$ are flagged as **Inconsistent / Out-of-Context Misinformation**.
4. **Classification Labels**:
   - `0`: Consistent / Real (Article and visual evidence match contextually)
   - `1`: Inconsistent / Fake (Article and image are mismatched, manipulated, or out-of-context)

---

## 🚀 Key Features

- **Interactive Streamlit Web Dashboard**: Test news claims and uploaded images with real-time gauges, visual domain detection, and detailed diagnostic explanations.
- **Calibrated Benchmark Presets**: Includes high-resolution test pairs across Politics, Disasters, Sports, and Technology.
- **RESTful API (FastAPI)**: Production-ready `/predict` and `/predict/text-only` endpoints with full telemetry and Pydantic validation.
- **End-to-End Test Suite**: Complete automated unit tests verifying inference, dataset integrity, and out-of-context sanity rules.

---

## 📂 Repository Structure

```text
FakeNews/
├── api/                  # FastAPI backend service
│   └── main.py
├── app.py                # Streamlit web dashboard
├── data/                 # Sample dataset and photo benchmarks
│   ├── images/           # Benchmark imagery and sample dataset
│   └── dataset.csv       # Multimodal metadata and labels
├── results/              # Evaluation metrics & confusion matrix
│   ├── confusion_matrix.png
│   ├── metrics_summary.json
│   └── training_history.json
├── saved_models/         # Serialized model weights & threshold configs
├── src/                  # Core source code
│   ├── dataset.py        # Dataset generation & CLIP feature extraction
│   ├── evaluate.py       # Benchmark evaluation & confusion matrix plotting
│   ├── models.py         # Calibrated detector & PyTorch architectures
│   └── train.py          # Training loop & checkpointing
├── tests/                # Automated test suite
│   ├── test_clip_calibration.py
│   ├── test_env.py
│   └── test_pipeline.py
├── requirements.txt      # Project dependencies
└── .gitignore            # Git exclusion rules
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- PyTorch (CUDA supported, CPU fallback included)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/shivanshg369/FakeNews.git
cd FakeNews

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Streamlit Web Application
```bash
streamlit run app.py
```

### 4. Running the FastAPI Backend
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API documentation is available at `http://localhost:8000/docs`.

### 5. Running the Automated Test Suite
```bash
# Run End-to-End Pipeline & QA tests
python -m unittest tests/test_pipeline.py

# Run Multimodal Calibration tests
python tests/test_clip_calibration.py

# Run Test Benchmark Evaluation
python -m src.evaluate
```

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
