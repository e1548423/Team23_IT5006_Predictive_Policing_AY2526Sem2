# =============================================================================
# FASTAPI BACKEND — Chicago Crime Prediction Model API
#
# Hosts the XGBoost model on Render.com and exposes a /predict endpoint
# that the Streamlit app calls instead of loading the model locally.
# =============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os

app = FastAPI(title="Chicago Crime Prediction API", version="1.0.0")

# ── Paths ────────────────────────────────────────────────────────────────────
DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "deployment")

# ── Load model at startup ────────────────────────────────────────────────────
pipeline = None
baselines = None
meta = None


@app.on_event("startup")
def load_model():
    global pipeline, baselines, meta
    pipeline = joblib.load(os.path.join(DEPLOY_DIR, "xgb_calibrated_pipeline.joblib"))
    baselines = pd.read_csv(os.path.join(DEPLOY_DIR, "tile_baseline.csv"))
    with open(os.path.join(DEPLOY_DIR, "metadata.json")) as f:
        meta = json.load(f)
    print(f"✓ Model loaded — {len(baselines)} tiles, ROC-AUC {meta['roc_auc']}")


# ── Request / Response schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    query_date: str          # "2026-03-17"
    shift: str               # "morning_noon" | "afternoon_night" | "overnight"
    threshold: float = 0.055
    live_lag: dict | None = None  # optional {h3_address: lag_1d_count, ...}


class TileResult(BaseModel):
    h3_address: str
    crime_probability: float
    flagged: int
    risk_tier: str
    shift: str
    query_date: str


class PredictResponse(BaseModel):
    results: list[TileResult]
    tile_count: int
    flagged_count: int


class MetadataResponse(BaseModel):
    roc_auc: float
    threshold: float
    precision: float | None = None
    recall: float | None = None
    tile_count: int
    trained_at: str
    feature_cols: list[str]


# ── Shift encoding ───────────────────────────────────────────────────────────
SHIFT_MAP = {
    "morning_noon":     {"is_afternoon_night": 0, "is_overnight": 0},
    "afternoon_night":  {"is_afternoon_night": 1, "is_overnight": 0},
    "overnight":        {"is_afternoon_night": 0, "is_overnight": 1},
}


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "tiles": len(baselines) if baselines is not None else 0}


@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    return MetadataResponse(
        roc_auc=meta["roc_auc"],
        threshold=meta["threshold"],
        precision=meta.get("precision"),
        recall=meta.get("recall"),
        tile_count=len(baselines),
        trained_at=meta.get("trained_at", "N/A"),
        feature_cols=meta["feature_cols"],
    )


class PRAtThresholdResponse(BaseModel):
    threshold: float
    precision: float
    recall: float


@app.get("/pr_at_threshold", response_model=PRAtThresholdResponse)
def pr_at_threshold(threshold: float):
    """Interpolate precision/recall from the saved PR curve for any threshold."""
    pr_curve = meta.get("pr_curve")
    if not pr_curve:
        raise HTTPException(404, "PR curve not saved in metadata. Retrain the model.")

    thresholds = np.array(pr_curve["thresholds"])
    precisions = np.array(pr_curve["precision"])
    recalls = np.array(pr_curve["recall"])

    # Clamp to curve range
    t = np.clip(threshold, thresholds.min(), thresholds.max())

    # Interpolate (thresholds are ascending from precision_recall_curve)
    prec = float(np.interp(t, thresholds, precisions))
    rec = float(np.interp(t, thresholds, recalls))

    return PRAtThresholdResponse(
        threshold=round(threshold, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
    )


@app.get("/baselines")
def get_baselines():
    """Return tile_baseline data so Streamlit can build the beat map without the model."""
    return {
        "h3_addresses": baselines["h3_address"].tolist(),
        "columns": baselines.columns.tolist(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.shift not in SHIFT_MAP:
        raise HTTPException(400, f"Invalid shift. Use: {list(SHIFT_MAP.keys())}")

    tiles = baselines.copy()
    feature_cols = meta["feature_cols"]

    # Apply live lag overrides if provided
    if req.live_lag:
        lag_df = pd.DataFrame(
            list(req.live_lag.items()), columns=["h3_address", "lag_1d_fresh"]
        )
        lag_df["lag_1d_fresh"] = pd.to_numeric(lag_df["lag_1d_fresh"], errors="coerce")
        tiles = tiles.merge(lag_df, on="h3_address", how="left")
        tiles["lag_1d"] = tiles["lag_1d_fresh"].fillna(tiles["lag_1d"])
        tiles.drop(columns=["lag_1d_fresh"], inplace=True)

    # Shift features
    s = SHIFT_MAP[req.shift]
    tiles["is_afternoon_night"] = s["is_afternoon_night"]
    tiles["is_overnight"] = s["is_overnight"]

    # Temporal features
    dt = pd.Timestamp(req.query_date)
    dow, mon = dt.dayofweek, dt.month
    tiles["day_sin"] = np.sin(2 * np.pi * dow / 7)
    tiles["day_cos"] = np.cos(2 * np.pi * dow / 7)
    tiles["month_sin"] = np.sin(2 * np.pi * (mon - 1) / 12)
    tiles["month_cos"] = np.cos(2 * np.pi * (mon - 1) / 12)

    # Predict
    X = tiles[feature_cols]
    probs = pipeline.predict_proba(X)[:, 1]

    results = tiles[["h3_address"]].copy()
    results["crime_probability"] = probs.round(4)
    results["flagged"] = (probs >= req.threshold).astype(int)
    results["risk_tier"] = pd.cut(
        probs, bins=[0, 0.10, 0.20, 0.35, 1.0],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)
    results["shift"] = req.shift
    results["query_date"] = req.query_date

    results = results.sort_values("crime_probability", ascending=False).reset_index(drop=True)

    return PredictResponse(
        results=[TileResult(**row) for row in results.to_dict("records")],
        tile_count=len(results),
        flagged_count=int(results["flagged"].sum()),
    )
