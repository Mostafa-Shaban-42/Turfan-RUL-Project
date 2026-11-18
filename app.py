"""Research dashboard - Full comparison for final demo - FIXED PATHS v2."""
from pathlib import Path
import json, joblib
import pandas as pd
import streamlit as st
try:
    import tensorflow as tf
except ImportError:
    tf = None
import numpy as np

ROOT = Path(__file__).resolve().parent

# ==================== FIX PATHS - Auto Detection ====================
def find_folder(candidates, required_file):
    """يبحث عن المجلد الصحيح تلقائيا"""
    for p in candidates:
        if (p / required_file).exists():
            return p
    # لو لم يجد، يرجع أول مسار فيه أي ملفات
    for p in candidates:
        if p.exists() and any(p.iterdir()):
            return p
    return candidates[0]

# احتمالات مكان مجلد TRAINED
TRAINED_CANDIDATES = [
    ROOT / "models" / "trained",
    ROOT / "trained",
    ROOT / "turfan" / "models" / "trained",
    ROOT / "turfan" / "trained",
    Path("/Users/mac/Projects/turfan/models/trained"),
    Path("/Users/mac/Projects/turfan/trained"),
]

# احتمالات مكان PROCESSED
PROCESSED_CANDIDATES = [
    ROOT / "data" / "processed",
    ROOT / "turfan" / "data" / "processed",
    ROOT / "models" / "trained",  # fallback
]

# احتمالات مكان SCALERS
SCALERS_CANDIDATES = [
    ROOT / "models" / "scalers",
    ROOT / "turfan" / "models" / "scalers",
    ROOT / "data" / "processed",
    ROOT / "models" / "trained",
]

PROCESSED = find_folder(PROCESSED_CANDIDATES, "metadata.json")
TRAINED = find_folder(TRAINED_CANDIDATES, "Table4_Main_Results_mean_sd.csv")
SCALERS = find_folder(SCALERS_CANDIDATES, "standard_scaler.joblib")

# لو PROCESSED لم يوجد فيه metadata، جرب TRAINED نفسه
if not (PROCESSED / "metadata.json").exists() and (TRAINED / "metadata.json").exists():
    PROCESSED = TRAINED

print(f"ROOT={ROOT}")
print(f"PROCESSED={PROCESSED} exists={PROCESSED.exists()}")
print(f"TRAINED={TRAINED} exists={TRAINED.exists()}")
print(f"SCALERS={SCALERS} exists={SCALERS.exists()}")

# ==================== Streamlit Config ====================
st.set_page_config(page_title="Turbofan RUL - Full Comparison", layout="wide")
st.title("Turbofan RUL - NASA FD001 - Full Model Comparison")
st.caption("Baseline vs Classic vs Modern vs Advanced - For PUA DL Final")

# Sidebar Debug
with st.sidebar:
    st.write("**Paths Debug:**")
    st.text(f"TRAINED: {TRAINED}\nExists: {TRAINED.exists()}")
    if TRAINED.exists():
        csvs = list(TRAINED.glob("*.csv"))
        keras_files = list(TRAINED.glob("*.keras"))
        st.text(f"{len(csvs)} CSV, {len(keras_files)} KERAS found")
        if csvs:
            st.text("CSV files:\n" + "\n".join([c.name for c in csvs[:5]]))

def load():
    # Load metadata
    meta_path = PROCESSED / "metadata.json"
    if not meta_path.exists():
        # حاول في TRAINED
        meta_path = TRAINED / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {PROCESSED} or {TRAINED}")
    
    meta = json.loads(meta_path.read_text())
    
    # Load scaler
    scaler_path = SCALERS / "standard_scaler.joblib"
    if not scaler_path.exists():
        scaler_path = PROCESSED / "standard_scaler.joblib"
    if not scaler_path.exists():
        scaler_path = TRAINED / "standard_scaler.joblib"
    
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        # لو مفيش scaler، هنشتغل بدون scaling في الديمو (للتجربة فقط)
        scaler = None
        
    return meta, scaler

try:
    meta, scaler = load()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.write(f"PROCESSED tried: {PROCESSED}")
    st.write(f"TRAINED tried: {TRAINED}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Benchmark Comparison", "Ablation & Error Analysis", "Single Engine Prediction"])

with tab1:
    try:
        table4_path = TRAINED / "Table4_Main_Results_mean_sd.csv"
        if not table4_path.exists():
            # دور في كل مكان
            for p in TRAINED_CANDIDATES:
                if (p / "Table4_Main_Results_mean_sd.csv").exists():
                    table4_path = p / "Table4_Main_Results_mean_sd.csv"
                    break
        
        main_df = pd.read_csv(table4_path)
        st.subheader("Table 4: Main Results (mean ± SD over 3 seeds)")
        st.dataframe(main_df, use_container_width=True)
        
        all_runs_path = TRAINED / "all_runs.csv"
        if not all_runs_path.exists():
            for p in TRAINED_CANDIDATES:
                if (p / "all_runs.csv").exists():
                    all_runs_path = p / "all_runs.csv"
                    break
        
        if all_runs_path.exists():
            all_runs = pd.read_csv(all_runs_path)
            st.line_chart(all_runs.pivot_table(index="seed", columns="model", values="rmse"))
        else:
            st.info(f"all_runs.csv not found, but Table4 loaded from {table4_path}")
            
    except Exception as e:
        st.error(f"Run 03 first: {e}")
        st.info(f"Looked in: {TRAINED}\nContent: {list(TRAINED.glob('*')) if TRAINED.exists() else 'Folder not exists'}")

with tab2:
    try:
        def load_csv(name):
            p = TRAINED / name
            if not p.exists():
                for cand in TRAINED_CANDIDATES:
                    if (cand / name).exists():
                        return pd.read_csv(cand / name)
                raise FileNotFoundError(name)
            return pd.read_csv(p)
        
        st.write("Table 5: Ablation Study")
        st.dataframe(load_csv("Table5_Ablation.csv"), use_container_width=True)
        
        st.write("Table 6: Error Analysis - Top 10 worst engines")
        st.dataframe(load_csv("Table6_Error_Analysis.csv"), use_container_width=True)
        
        st.write("Table 7: Computational Cost")
        st.dataframe(load_csv("Table7_Computational_Cost.csv"), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading tables: {e}")

with tab3:
    if tf is None:
        st.warning("TensorFlow not installed - prediction disabled. Run: pip install tensorflow")
    else:
        f = st.file_uploader("Upload engine CSV (26 cols, no header)", type=["csv","txt"])
        model_choice = st.selectbox("Choose model", ["cnn_lstm_attention","transformer","cnn_lstm","lstm","baseline_mlp"])
        seed_choice = st.selectbox("Seed", [42,123,2026])
        if f:
            try:
                df = pd.read_csv(f, sep=r"\s+|,", header=None, engine="python")
                if df.shape[1] != 26:
                    raise ValueError(f"26 cols expected, got {df.shape[1]}")
                df.columns = meta["input_columns"]
                df = df.apply(pd.to_numeric, errors="coerce")
                feats = meta["input_features"]
                win = meta["window_size"]
                
                if scaler is not None:
                    sc = scaler.transform(df[feats]).astype(np.float32)
                else:
                    sc = df[feats].to_numpy(np.float32)
                
                if len(sc) < win:
                    raise ValueError(f"Need {win} cycles, got {len(sc)}")
                
                model_path = TRAINED / f"{model_choice}_seed{seed_choice}.keras"
                if not model_path.exists():
                    for cand in TRAINED_CANDIDATES:
                        if (cand / f"{model_choice}_seed{seed_choice}.keras").exists():
                            model_path = cand / f"{model_choice}_seed{seed_choice}.keras"
                            break
                
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                model = tf.keras.models.load_model(model_path)
                pred = float(model.predict(sc[-win:][None,...], verbose=0).reshape(-1)[0])
                st.metric("Predicted RUL", f"{max(0,pred):.1f} cycles")
                st.success(f"Loaded: {model_path.name}")
            except Exception as e:
                st.error(f"Prediction error: {e}")