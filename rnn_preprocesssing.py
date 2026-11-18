"""Create fixed-length RUL sequences for CNN-LSTM training and FD001 testing."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
SEQUENCES = ROOT / "data" / "sequences"
SEQUENCES.mkdir(parents=True, exist_ok=True)
WINDOW_SIZE = 30
RUL_CAP = 125

def load_metadata():
    return json.loads((PROCESSED/"metadata.json").read_text())

def build_training_windows(df, features, window, cap):
    xs, ys = [], []
    for _, group in df.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        vals = group[features].to_numpy(np.float32)
        cycles = group["cycle"].to_numpy(np.int64)
        max_c = int(cycles.max())
        if len(vals)<window: raise ValueError("too short")
        rul = np.minimum(max_c - cycles, cap).astype(np.float32)
        for end in range(window, len(vals)+1):
            xs.append(vals[end-window:end]); ys.append(rul[end-1])
    return np.asarray(xs,np.float32), np.asarray(ys,np.float32)

def build_test_windows(df, features, window):
    xs, ids = [], []
    for eid, group in df.groupby("engine_id", sort=True):
        group=group.sort_values("cycle")
        vals=group[features].to_numpy(np.float32)
        if len(vals)<window: raise ValueError("test too short")
        xs.append(vals[-window:]); ids.append(int(eid))
    return np.asarray(xs,np.float32), np.asarray(ids,np.int64)

def main():
    meta=load_metadata(); feats=meta["input_features"]; win=meta["window_size"]; cap=meta["rul_cap"]
    arrs={}
    for split in ("train","validation"):
        df=pd.read_csv(PROCESSED/f"{split}_scaled.csv")
        x,y=build_training_windows(df,feats,win,cap)
        arrs[f"X_{split}"]=x; arrs[f"y_{split}"]=y
    test_df=pd.read_csv(PROCESSED/"test_scaled.csv")
    x_test,ids=build_test_windows(test_df,feats,win)
    y_test=np.load(PROCESSED/"official_test_rul.npy").astype(np.float32)
    arrs.update({"X_test":x_test,"y_test":y_test,"test_engine_ids":ids})
    for k,v in arrs.items(): np.save(SEQUENCES/f"{k}.npy", v)
    (SEQUENCES/"sequence_manifest.json").write_text(json.dumps({"experiment_id":meta["experiment_id"],"window_size":win,"rul_cap":cap,"input_features":feats,"shapes":{k:list(v.shape) for k,v in arrs.items()}},indent=2))
    print("Sequences done")

if __name__=="__main__": main()