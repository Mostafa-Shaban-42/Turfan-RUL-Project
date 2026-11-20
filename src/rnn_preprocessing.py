"""Create fixed-length RUL sequences for CNN-LSTM training and FD001 testing.
Run: python3 -u 02_rnn_preprocessing.py
Must run 01_data_preprocessing.py first.
"""
from __future__ import annotations
import json
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
    meta_path = PROCESSED / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {PROCESSED}. Run 01_data_preprocessing.py first."
        )
    return json.loads(meta_path.read_text())

def build_training_windows(df, features, window, cap):
    xs, ys = [], []
    for _, group in df.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        vals = group[features].to_numpy(np.float32)
        cycles = group["cycle"].to_numpy(np.int64)
        max_c = int(cycles.max())
        if len(vals) < window:
            raise ValueError(f"Engine too short: {len(vals)} < {window}")
        rul = np.minimum(max_c - cycles, cap).astype(np.float32)
        for end in range(window, len(vals) + 1):
            xs.append(vals[end - window:end])
            ys.append(rul[end - 1])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)

def build_test_windows(df, features, window):
    xs, ids = [], []
    for eid, group in df.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        vals = group[features].to_numpy(np.float32)
        if len(vals) < window:
            raise ValueError(f"Test engine {eid} too short: {len(vals)} < {window}")
        xs.append(vals[-window:])
        ids.append(int(eid))
    return np.asarray(xs, np.float32), np.asarray(ids, np.int64)

def main():
    meta = load_metadata()
    feats = meta["input_features"]
    win = meta["window_size"]
    cap = meta["rul_cap"]
    arrs = {}
    for split in ("train", "validation"):
        csv_path = PROCESSED / f"{split}_scaled.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found. Run 01 first.")
        df = pd.read_csv(csv_path)
        x, y = build_training_windows(df, feats, win, cap)
        arrs[f"X_{split}"] = x
        arrs[f"y_{split}"] = y
        print(f"{split}: X={x.shape} y={y.shape}")
    test_df = pd.read_csv(PROCESSED / "test_scaled.csv")
    x_test, ids = build_test_windows(test_df, feats, win)
    y_test = np.load(PROCESSED / "official_test_rul.npy").astype(np.float32)
    arrs.update({"X_test": x_test, "y_test": y_test, "test_engine_ids": ids})
    print(f"test: X={x_test.shape} y={y_test.shape} ids={ids.shape}")
    for k, v in arrs.items():
        np.save(SEQUENCES / f"{k}.npy", v)
    manifest = {
        "experiment_id": meta["experiment_id"],
        "window_size": win,
        "rul_cap": cap,
        "input_features": feats,
        "shapes": {k: list(v.shape) for k, v in arrs.items()}
    }
    (SEQUENCES / "sequence_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Sequences done - saved to", SEQUENCES)

if __name__ == "__main__":
    main()