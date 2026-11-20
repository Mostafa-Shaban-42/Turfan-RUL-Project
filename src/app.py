"""Research dashboard - Full comparison for final demo - FIXED FOR turfan/src layout.
Run: python3 -m streamlit run src/app.py  (from /Users/mac/Projects/turfan)
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
try:
    import tensorflow as tf
except ImportError:
    tf = None
import numpy as np

# FIX: app.py جوه src/ يبقى ROOT هو turfan/ مش turfan/src/
ROOT = Path(__file__).resolve().parent.parent

# الأماكن المحتملة - بيجربهم كلهم
PROCESSED_CANDIDATES = [
    ROOT / "data" / "processed",
    ROOT / "src" / "data" / "processed",
    ROOT,
]
TRAINED_CANDIDATES = [
    ROOT / "trained",
    ROOT / "models" / "trained",
    ROOT / "src" / "trained",
    ROOT,
]
SCALER_CANDIDATES = [
    ROOT / "models" / "scalers",
    ROOT / "data" / "processed",
    ROOT,
]

def pick_first(candidates, filename):
    for c in candidates:
        if (c / filename).exists():
            return c
    return candidates[0]

PROCESSED = pick_first(PROCESSED_CANDIDATES, "metadata.json")
TRAINED = pick_first(TRAINED_CANDIDATES, "Table4_Main_Results_mean_sd.csv")
SCALERS = pick_first(SCALER_CANDIDATES, "standard_scaler.joblib")

st.set_page_config(page_title="Turbofan RUL - Full Comparison", layout="wide")
st.title("Turbofan RUL - NASA FD001 - Full Model Comparison")
st.caption("Baseline vs Classic vs Modern vs Advanced - For PUA DL Final")

with st.sidebar:
    st.write("**Paths Debug:**")
    st.text(f"ROOT: {ROOT}\nPROCESSED: {PROCESSED}\nTRAINED: {TRAINED}")
    if TRAINED.exists():
        csvs = list(TRAINED.glob("*.csv"))
        keras_files = list(TRAINED.glob("*.keras"))
        st.text(f"{len(csvs)} CSV, {len(keras_files)} KERAS found")

def load():
    meta_path = PROCESSED / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {PROCESSED}. Run 01 first.")
    meta = json.loads(meta_path.read_text())
    scaler_path = SCALERS / "standard_scaler.joblib"
    if not scaler_path.exists():
        scaler_path = PROCESSED / "standard_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return meta, scaler

try:
    meta, scaler = load()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Benchmark Comparison", "Ablation & Error Analysis", "Single Engine Prediction"])

with tab1:
    try:
        main_df = pd.read_csv(TRAINED / "Table4_Main_Results_mean_sd.csv")
        st.subheader("Table 4: Main Results (mean ± SD over 3 seeds)")
        st.dataframe(main_df, use_container_width=True)
        if (TRAINED / "all_runs.csv").exists():
            all_runs = pd.read_csv(TRAINED / "all_runs.csv")
            st.line_chart(all_runs.pivot_table(index="seed", columns="model", values="rmse"))
    except Exception as e:
        st.error(f"Run 03 first: {e}")

with tab2:
    try:
        st.write("Table 5: Ablation Study")
        st.dataframe(pd.read_csv(TRAINED / "Table5_Ablation.csv"), use_container_width=True)
        st.write("Table 6: Error Analysis - Top 10 worst engines (Best Model)")
        st.dataframe(pd.read_csv(TRAINED / "Table6_Error_Analysis.csv"), use_container_width=True)
        st.write("Table 7: Computational Cost (Real Measured)")
        st.dataframe(pd.read_csv(TRAINED / "Table7_Computational_Cost.csv"), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading tables: {e}")

with tab3:
    if tf is None:
        st.warning("TensorFlow not installed - prediction disabled. Run: pip install tensorflow")
    else:
        f = st.file_uploader("Upload engine CSV (26 cols, no header)", type=["csv", "txt"])
        model_choice = st.selectbox("Choose model", ["cnn_lstm_attention", "transformer", "cnn_lstm", "lstm", "baseline_mlp"])
        seed_choice = st.selectbox("Seed", [42, 123, 2026])
        if f:
            try:
                df = pd.read_csv(f, sep=r"\s+|,", header=None, engine="python")
                if df.shape[1] != 26:
                    raise ValueError(f"26 cols expected, got {df.shape[1]}")
                df.columns = meta["input_columns"]
                df = df.apply(pd.to_numeric, errors="coerce")
                feats = meta["input_features"]
                win = meta["window_size"]
                sc = scaler.transform(df[feats]).astype(np.float32) if scaler is not None else df[feats].to_numpy(np.float32)
                if len(sc) < win:
                    raise ValueError(f"Need {win} cycles, got {len(sc)}")
                model_path = TRAINED / f"{model_choice}_seed{seed_choice}.keras"
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                model = tf.keras.models.load_model(model_path)
                pred = float(model.predict(sc[-win:][None, ...], verbose=0).reshape(-1)[0])
                st.metric("Predicted RUL", f"{max(0, pred):.1f} cycles")
                st.success(f"Loaded: {model_path.name}")
            except Exception as e:
                st.error(f"Prediction error: {e}")