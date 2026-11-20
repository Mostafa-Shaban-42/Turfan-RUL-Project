"""
Download, validate, split and scale NASA C-MAPSS FD001 data.
Dataset: NASA C-MAPSS FD001
Run: python3 -u 01_data_preprocessing.py
"""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SCALERS = ROOT / "models" / "scalers"
for directory in (RAW, PROCESSED, SCALERS):
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================
BASE_URL = "https://raw.githubusercontent.com/SumantMalekar/CMAPSSData/main/"
FILES = {"train": "train_FD001.txt", "test": "test_FD001.txt", "rul": "RUL_FD001.txt"}
COLUMNS = (["engine_id", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)])
RANDOM_SEED = 42
VALIDATION_SIZE = 0.20
VARIANCE_THRESHOLD = 1e-3
WINDOW_SIZE = 30
RUL_CAP = 125
DOWNLOAD_TIMEOUT = 60

# ============================================================
# 3. LOGGING
# ============================================================
def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

# ============================================================
# 4. SHA256 CHECKSUM
# ============================================================
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# ============================================================
# 5. DOWNLOAD DATASET FILES
# ============================================================
def download_file(filename: str) -> Path:
    dest = RAW / filename
    if dest.exists() and dest.stat().st_size > 100:
        log(f"[EXISTS] {filename}")
        return dest
    url = BASE_URL + filename
    log(f"[DOWNLOAD] {filename}")
    log(f"[URL] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 TurbofanRUL/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP status: {response.status}")
            data = response.read()
        if not data or len(data) < 100:
            raise ValueError(f"Downloaded file is empty or too small: {filename}")
        lower_start = data[:1000].lower()
        if b"<html" in lower_start or b"<!doctype" in lower_start:
            raise ValueError(f"HTML received instead of dataset: {filename}")
        fd, temporary_path = tempfile.mkstemp(dir=RAW, prefix=f".{filename}.")
        os.close(fd)
        temporary_file = Path(temporary_path)
        try:
            temporary_file.write_bytes(data)
            temporary_file.replace(dest)
        finally:
            if temporary_file.exists():
                temporary_file.unlink()
        log(f"[SUCCESS] {filename} ({len(data):,} bytes)")
        return dest
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP error downloading {filename}: {error.code} {error.reason}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Network error downloading {filename}: {error.reason}") from error
    except Exception as error:
        raise RuntimeError(f"Failed downloading {filename}: {error}") from error

# ============================================================
# 6. LOAD AND VALIDATE C-MAPSS DATA
# ============================================================
def read_cmapss(path: Path) -> pd.DataFrame:
    log(f"[READ] {path.name}")
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != 26:
        raise ValueError(f"{path.name}: Expected 26 columns, received {df.shape[1]}")
    df.columns = COLUMNS
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"{path.name}: Dataset contains NaN values")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{path.name}: Dataset contains infinite values")
    for column in ("engine_id", "cycle"):
        if (numeric[column] <= 0).any():
            raise ValueError(f"{path.name}: Invalid {column}")
        numeric[column] = numeric[column].astype(np.int64)
    if numeric.duplicated(["engine_id", "cycle"]).any():
        raise ValueError(f"{path.name}: Duplicate engine/cycle records")
    for engine_id, group in numeric.groupby("engine_id"):
        cycles = group["cycle"].to_numpy()
        if len(cycles) > 1 and not np.all(np.diff(cycles) > 0):
            raise ValueError(f"Engine {engine_id}: Cycles are not strictly increasing")
    log(f"[VALIDATED] {path.name}: {len(numeric):,} rows, {numeric['engine_id'].nunique()} engines")
    return numeric

# ============================================================
# 7. LOAD OFFICIAL TEST RUL
# ============================================================
def read_official_rul(path: Path, expected_engines: int) -> np.ndarray:
    log("[READ] Official test RUL")
    values = pd.read_csv(path, sep=r"\s+", header=None)
    if values.shape[1] != 1:
        raise ValueError("Official RUL file must contain one column")
    y = pd.to_numeric(values.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    if len(y) != expected_engines:
        raise ValueError(f"RUL count mismatch: expected {expected_engines}, received {len(y)}")
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Official RUL contains invalid values")
    log(f"[VALIDATED] Official RUL: {len(y)} engines")
    return y

# ============================================================
# 8. FEATURE SELECTION
# ============================================================
def select_features(train_df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    log("[FEATURE SELECTION] Calculating sensor variances")
    variances = train_df[feature_columns].var(ddof=0)
    selected = variances[variances > VARIANCE_THRESHOLD].index.tolist()
    if not selected:
        raise ValueError("Feature selection returned zero features")
    removed = sorted(set(feature_columns) - set(selected))
    log(f"[FEATURE SELECTION] Selected {len(selected)}/{len(feature_columns)} features")
    if removed:
        log(f"[REMOVED FEATURES] {removed}")
    return selected

# ============================================================
# 9. SCALING AND EXPORT
# ============================================================
def scale_and_export(scaler: StandardScaler, frame: pd.DataFrame, features: list[str], output_name: str) -> None:
    log(f"[SCALING] {output_name}")
    scaled = scaler.transform(frame[features]).astype(np.float32)
    output = frame[["engine_id", "cycle"]].copy()
    for index, feature in enumerate(features):
        output[feature] = scaled[:, index]
    output_path = PROCESSED / f"{output_name}_scaled.csv"
    output.to_csv(output_path, index=False)
    log(f"[SAVED] {output_path.name} ({len(output):,} rows)")

# ============================================================
# 10. MAIN PIPELINE
# ============================================================
def main():
    start_time = datetime.now(timezone.utc)
    log("=" * 65)
    log("NASA C-MAPSS FD001 DATA PREPROCESSING")
    log("=" * 65)
    log("[STEP 1/8] Downloading dataset")
    paths = {key: download_file(filename) for key, filename in FILES.items()}
    log("[STEP 2/8] Loading and validating dataset")
    train = read_cmapss(paths["train"])
    test = read_cmapss(paths["test"])
    test_ids = sorted(test["engine_id"].unique())
    rul = read_official_rul(paths["rul"], len(test_ids))
    log("[STEP 3/8] Splitting training engines")
    train_ids = np.array(sorted(train["engine_id"].unique()))
    tr_ids, val_ids = train_test_split(train_ids, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
    tr = train[train["engine_id"].isin(tr_ids)].copy()
    va = train[train["engine_id"].isin(val_ids)].copy()
    log(f"[SPLIT] Training engines: {len(tr_ids)}")
    log(f"[SPLIT] Validation engines: {len(val_ids)}")
    log(f"[SPLIT] Test engines: {len(test_ids)}")
    log("[STEP 4/8] Feature selection")
    feature_columns = [c for c in COLUMNS if c not in ("engine_id", "cycle")]
    selected_features = select_features(tr, feature_columns)
    log("[STEP 5/8] Fitting StandardScaler")
    scaler = StandardScaler()
    scaler.fit(tr[selected_features])
    log("[STEP 6/8] Scaling and exporting datasets")
    scale_and_export(scaler, tr, selected_features, "train")
    scale_and_export(scaler, va, selected_features, "validation")
    scale_and_export(scaler, test, selected_features, "test")
    log("[STEP 7/8] Saving preprocessing artifacts")
    scaler_path = SCALERS / "standard_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    np.save(PROCESSED / "official_test_rul.npy", rul)
    log("[STEP 8/8] Creating experiment metadata")
    experiment_time = datetime.now(timezone.utc)
    metadata = {
        "dataset": "NASA C-MAPSS FD001",
        "experiment_id": experiment_time.strftime("fd001_%Y%m%dT%H%M%SZ"),
        "created_utc": experiment_time.isoformat(),
        "random_seed": RANDOM_SEED,
        "validation_size": VALIDATION_SIZE,
        "variance_threshold": VARIANCE_THRESHOLD,
        "input_columns": COLUMNS,
        "input_features": selected_features,
        "window_size": WINDOW_SIZE,
        "rul_cap": RUL_CAP,
        "train_engine_ids": sorted(map(int, tr_ids)),
        "validation_engine_ids": sorted(map(int, val_ids)),
        "test_engine_ids": list(map(int, test_ids)),
        "statistics": {"train_rows": len(tr), "validation_rows": len(va), "test_rows": len(test), "train_engines": len(tr_ids), "validation_engines": len(val_ids), "test_engines": len(test_ids), "selected_features_count": len(selected_features), "original_features_count": len(feature_columns)},
        "hypotheses": ["H1: CNN-LSTM outperforms LSTM due to local sensor features", "H2: Transformer with attention outperforms RNNs on long degradation", "H3: CNN-LSTM+Attention reduces NASA score vs CNN-LSTM", "H4: Jitter augmentation improves generalization"],
        "source_files": {key: {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)} for key, path in paths.items()}
    }
    metadata_path = PROCESSED / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    log("=" * 65)
    log("PREPROCESSING COMPLETED SUCCESSFULLY")
    log("=" * 65)
    log(f"Dataset: NASA C-MAPSS FD001")
    log(f"Selected features: {len(selected_features)}")
    log(f"Training engines: {len(tr_ids)}")
    log(f"Validation engines: {len(val_ids)}")
    log(f"Test engines: {len(test_ids)}")
    log(f"Processing time: {elapsed:.2f} seconds")
    log(f"Raw directory: {RAW}")
    log(f"Processed directory: {PROCESSED}")
    log(f"Scaler directory: {SCALERS}")
    log("=" * 65)
    log("Generated files:")
    for file in sorted(PROCESSED.iterdir()):
        if file.is_file():
            log(f"  - {file.name}")
    log(f"  - {scaler_path.name}")
    log("=" * 65)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("[INTERRUPTED] Processing stopped by user")
        raise SystemExit(1)
    except Exception as error:
        log(f"[FATAL ERROR] {error}")
        raise