# Turfan RUL Project: NASA C-MAPSS Deep Learning & Remaining Useful Life (RUL) Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, end-to-end Machine Learning and Deep Learning pipeline designed to predict the **Remaining Useful Life (RUL)** of turbofan engines using the benchmark **NASA C-MAPSS (FD001)** dataset. This repository benchmarks multiple state-of-the-art neural architectures—ranging from Baseline MLPs and LSTMs to Hybrid CNN-LSTMs, Transformers, and Attention-augmented networks—tailored for robust Predictive Maintenance (PdM).

---

## 🏗️ System Architecture & Engineering Pipeline

```text
Turfan-RUL-Project/
│
├── data_preprocessing.py        # Automated ingestion, validation, variance-based feature selection, and scaling
├── rnn_preprocessing.py         # Temporal sliding-window sequence generation (Window Size = 30, RUL Capped = 125)
├── train_evaluate.py            # Multi-seed training framework (Seeds: 42, 123, 2026) with ablation study & error metrics
├── app.py                       # Interactive Streamlit Research Dashboard & Real-Time Inference Engine
├── README.md                    # Comprehensive System Documentation
├── data/
│   ├── raw/                     # Raw NASA C-MAPSS FD001 telemetry text files
│   ├── processed/               # Standardized scaled CSVs and experiment metadata (JSON)
│   └── sequences/               # NumPy array tensors (.npy) optimized for rapid training
└── models/
    ├── scalers/                 # Persistent sklearn StandardScaler artifacts (.joblib)
    └── trained/                 # Model checkpoints (.keras) and analytical performance reports (.csv)

🔬 Methodology & Model Benchmarking
To ensure rigorous evaluation and avoid single-seed bias, all models are evaluated across 3 distinct random seeds (42, 123, 2026) with early stopping, model checkpointing, and adaptive learning rate reduction (ReduceLROnPlateau).
Baseline MLP: Fully connected multi-layer perceptron acting as a performance baseline.
LSTM: Deep Recurrent Neural Network capturing temporal degradation dependencies.
CNN-LSTM: Hybrid architecture leveraging 1D Convolutions for local spatial feature extraction followed by LSTMs.
Transformer: Self-attention mechanism combined with learned positional encodings for long-range sequence modeling.
CNN-LSTM + Attention: Advanced hybrid framework integrating soft-attention mechanisms over convolutional-recurrent representations.
🚀 Quick Start & Installation
Prerequisites
Ensure you have Python 3.10 or higher installed. Clone the repository and install dependencies:
# 1. Clone the repository
git clone [https://github.com/Mostafa-Shaban-42/Turfan-RUL-Project.git](https://github.com/Mostafa-Shaban-42/Turfan-RUL-Project.git)
cd Turfan-RUL-Project

# 2. Install required packages
pip install tensorflow streamlit pandas numpy scikit-learn joblib

Execution Pipeline
Run the modular pipeline sequentially to download data, engineer features, train models, and generate evaluation tables:
# Step 1: Ingest, validate, clean, and scale the NASA FD001 dataset
python3 -u data_preprocessing.py

# Step 2: Generate fixed-length temporal sliding windows for training/testing
python3 rnn_preprocessing.py

# Step 3: Train all model configurations across multi-seeds and export metrics
python3 train_evaluate.py

Interactive Research Dashboard
Launch the Streamlit dashboard to visualize comparative benchmark results, ablation studies, and perform real-time single-engine RUL inference:
streamlit run app.py
(Access the local UI via your browser at http://localhost:8501)

📊 Performance & Analytical Artifacts
The training pipeline automatically generates structured artifacts inside models/trained/ compliant with rigorous reporting standards:
Table 4: Main benchmark results reporting 
Mean
±
Standard Deviation
Mean±Standard Deviation for MAE, RMSE, and NASA Scoring Metric across seeds.
Table 5: Ablation study quantifying performance impact upon component removal (e.g., Attention / Conv1D modules).
Table 6: Error analysis highlighting worst-performing engines for model robustness auditing.
Table 7: Computational cost profiles (parameter sizing and training runtime benchmarks).
