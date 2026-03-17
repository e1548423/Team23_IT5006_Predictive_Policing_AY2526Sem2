# Chicago Violent Crime Prediction — Patrol Dispatch Dashboard

Streamlit web app for real-time crime prediction and patrol dispatch planning in Chicago.

## Architecture

- **Model**: Calibrated XGBoost trained on 3 years of violent crime data (BATTERY, ASSAULT, ROBBERY)
- **Spatial grid**: H3 resolution-8 hexagonal tiles (~0.7 km²)
- **Features**: Temporal lags, EWMA momentum, city-normalised rolling stats, spatial neighbour spillover, cyclical time encoding
- **Inference**: Pre-trained model scores all tiles for any date × shift combination in <1 second

## Setup

### 1. Train the model (locally, in Jupyter)

Run `Inference_Engine_UI_v2.ipynb` STEP 0 to fetch data and train. This creates:

```
deployment/
├── xgb_calibrated_pipeline.joblib   # Calibrated XGBoost pipeline
├── tile_baseline.csv                # Per-tile feature baselines (851 tiles)
└── metadata.json                    # Model config + performance metrics
```

### 2. Deploy to Streamlit Cloud

1. Push this repo to GitHub (including the `deployment/` folder)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `streamlit_app.py`
5. Deploy

### 3. Update the model

Re-run STEP 0 in the notebook locally, then push the updated `deployment/` folder to GitHub. Streamlit Cloud will auto-redeploy.

## Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `deployment/` | Pre-trained model artefacts |
| `Inference_Engine_UI_v2.ipynb` | Training + full inference notebook |
