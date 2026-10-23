# 🔍 Multimodal Fake News Detection

A multimodal deep learning system designed to detect inconsistent or misleading news stories by analyzing cross-modal relationships between article text and accompanying images.

---

## 📌 Architecture Overview

The system uses a multimodal architecture that pairs a frozen **CLIP** (`openai/clip-vit-base-patch32`) backbone with a trainable **Multilayer Perceptron (MLP)** classifier:

1. **Feature Extraction**:
   - Frozen CLIP extracts L2-normalized text features ($512$-d) and image features ($512$-d).
   - Computes cross-modal cosine similarity ($1$-d) between normalized image and text embeddings.
2. **Feature Fusion**:
   - Concatenates `[text_features (512) | image_features (512) | cosine_similarity (1)]` $\rightarrow$ $1025$-d fused vector.
3. **Classification Head**:
   - Trainable MLP: `Linear(1025 -> 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(256 -> 1)`.
   - Sigmoid activation yields probability of cross-modal inconsistency / fake news.
4. **Classification Labels**:
   - `0`: Consistent / Real (Article and image match contextually)
   - `1`: Inconsistent / Fake (Article and image are mismatched or manipulated)

---

## 🚀 Key Features

- **Interactive Streamlit Web Application**: Modern UI for testing news articles, image uploads, visual inspection, and consistency score gauges.
- **RESTful API (FastAPI)**: Production-ready endpoints for single-item and batch multimodal predictions.
- **Pre-computed Feature Caching**: Accelerates dataset training by caching frozen CLIP embeddings.
- **Comprehensive Evaluation**: Metrics reporting including Accuracy, Precision, Recall, F1 score, and Confusion Matrix generation.

---

## 📂 Repository Structure

```text
FakeNews/
├── api/                  # FastAPI backend service
│   └── main.py
├── app.py                # Streamlit web dashboard
├── data/                 # Sample dataset and imagery
│   ├── images/           # Sample image dataset
│   └── dataset.csv       # Multimodal metadata and labels
├── results/              # Evaluation metrics & confusion matrix
│   ├── confusion_matrix.png
│   ├── metrics_summary.json
│   └── training_history.json
├── saved_models/         # Serialized model weights (best_model.pth)
├── src/                  # Core source code
│   ├── dataset.py        # Custom Dataset & CLIP feature extraction
│   ├── evaluate.py       # Evaluation and metric calculation
│   ├── models.py         # PyTorch multimodal architecture
│   └── train.py          # Training loop & checkpointing
├── tests/                # Unit & integration test suite
├── requirements.txt      # Project dependencies
└── .gitignore            # Git exclusion rules
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- PyTorch with CUDA support (optional, CPU supported)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/shivanshg369/FakeNews.git
cd FakeNews

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
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
Interactive API documentation will be available at `http://localhost:8000/docs`.

### 5. Training and Evaluation
```bash
# Train the classification head
python -m src.train

# Evaluate on test split
python -m src.evaluate
```

---

## 🧪 Testing
Run the test suite using pytest:
```bash
pytest tests/ -v
```

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
