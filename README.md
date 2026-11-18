# Turbofan RUL - NASA C-MAPSS FD001

**Deep Learning Final Project**

> Research-Grade System - 5 Models × 3 Seeds - Full Compliance

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![Status: Complete](https://img.shields.io/badge/Status-Complete-green)](#results)

---

## 📁 Actual Project Tree (Your Current Structure)
turfan/
├── data/
│   ├── raw/                              # 3 files - Original NASA data
│   │   ├── train_FD001.txt               # 20,631 rows × 26 cols - 100 engines
│   │   ├── test_FD001.txt                # 13,096 rows × 26 cols - 100 engines
│   │   └── RUL_FD001.txt                 # 100 rows - Official Ground Truth
│   ├── processed/                        # 5 files - Scaled + Metadata
│   │   ├── train_scaled.csv
│   │   ├── validation_scaled.csv
│   │   ├── test_scaled.csv
│   │   ├── metadata.json                 # experiment_id, hypotheses H1-H4, sha256
│   │   └── official_test_rul.npy
│   └── sequences/                        # 8 files - Windowed arrays (window=30)
│       ├── X_train.npy, y_train.npy
│       ├── X_validation.npy, y_validation.npy
│       ├── X_test.npy, y_test.npy
│       ├── test_engine_ids.npy
│       └── sequence_manifest.json
├── models/
│   └── scalers/
│       └── standard_scaler.joblib        # StandardScaler fit on train only
├── trained/                              # 19 files - 15 models + 4 tables + logs
│   ├── baseline_mlp_seed42.keras
│   ├── baseline_mlp_seed123.keras
│   ├── baseline_mlp_seed2026.keras
│   ├── lstm_seed42.keras
│   ├── lstm_seed123.keras
│   ├── lstm_seed2026.keras
│   ├── cnn_lstm_seed42.keras
│   ├── cnn_lstm_seed123.keras
│   ├── cnn_lstm_seed2026.keras
│   ├── transformer_seed42.keras
│   ├── transformer_seed123.keras
│   ├── transformer_seed2026.keras
│   ├── cnn_lstm_attention_seed42.keras   # BEST MODEL
│   ├── cnn_lstm_attention_seed123.keras
│   ├── cnn_lstm_attention_seed2026.keras
│   ├── Table4_Main_Results_mean_sd.csv    # REQUIRED - Main results mean±SD
│   ├── Table5_Ablation.csv               # REQUIRED - Ablation Study
│   ├── Table6_Error_Analysis.csv         # REQUIRED - Top 10 worst engines
│   ├── Table7_Computational_Cost.csv      # REQUIRED - Params & Time
│   ├── all_runs.csv                      # All 15 runs detailed
│   └── metrics.json
├── data_preprocessing.py                 # File 1: Download + Validate + Split + Scale
├── rnn_preprocessing.py                  # File 2: Build 30-window sequences
├── train_evaluate.py                     # File 3: Train 5×3=15 models + Tables
├── app.py                                # File 4: Streamlit Full Comparison Demo
├── rnn_preprocesssing.py                 # DUPLICATE - Delete this (typo)
├── README.md                             # This file
└── .DS_Store                             # Mac system file - ignore

---

## 1. Dataset Details

**NASA C-MAPSS FD001 - Turbofan Engine Degradation**

- **What it is:** Simulated 100 engines run to failure, 100 engines truncated
- **Rows:** Train 20,631, Test 13,096, RUL 100
- **26 Columns:** `engine_id`, `cycle`, `setting_1-3`, `sensor_1-21`
- **Sensors:** Temperatures T2,T24,T30,T50, Pressures P2,P15,P30, Fan speeds Nf,Nc, etc.

**Preprocessing Logic (data_preprocessing.py):**
- Download from `SumantMalekar/CMAPSSData` with HTML check `<html` + `<!doctype`
- Validate: 26 cols, numeric, finite, no NaN, no duplicates, cycles strictly increasing
- **Engine-Level Split (Critical):** 80 train engines, 20 val engines, 100 test engines separate - prevents leakage
- Variance filter `>1e-3` → ~17 effective features from 21 sensors
- Scaler fit on train only → `models/scalers/standard_scaler.joblib`

---

## 2. Input / Output

- **Input:** Window `X ∈ R^(30 × F)` where F≈17 selected sensors, last 30 cycles
- **Output:** Scalar RUL `y ∈ R+` capped at 125 (standard in literature)
- **Example:** Engine 5 last 30 cycles → Model predicts `28.5 cycles` remaining

---

## 3. Installation & Quick Fix

Your current tree has 2 issues to fix for 100% mark:

```bash
cd /Users/mac/Projects/turfan  # your path from tree

# 1. Delete duplicate typo file
rm -f rnn_preprocesssing.py

# 2. Fix structure to match spec 4.1 (trained should be inside models/)
mkdir -p models/trained
cp trained/* models/trained/ 2>/dev/null
# Keep both for safety, app.py will auto-detect

# 3. Install deps
pip install numpy pandas scikit-learn tensorflow joblib streamlit

# 4. Run demo
streamlit run app.py
streamlit run app.py
# Open http://localhost:8501

Full pipeline from scratch
python data_preprocessing.py   # Step 1: 30 sec
python rnn_preprocessing.py    # Step 2: 10 sec
python train_evaluate.py       # Step 3: 60-120 min for 15 models
streamlit run app.py           # Step 4
