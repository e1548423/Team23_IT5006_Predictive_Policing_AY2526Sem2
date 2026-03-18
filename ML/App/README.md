# Chicago Violent Crime Prediction — Patrol Dispatch Dashboard

Streamlit web app for real-time crime prediction and patrol dispatch planning in Chicago, powered by a FastAPI model backend hosted on Render.

## Folder Structure

```
ML/
├── App/                        ← Streamlit Cloud (UI only)
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── README.md
└── Deploy_Render/              ← Render.com (FastAPI model API)
    ├── main.py
    ├── requirements.txt
    ├── render.yaml
    └── deployment/
        ├── xgb_calibrated_pipeline.joblib
        ├── tile_baseline.csv
        └── metadata.json
```

The model runs on Render as a FastAPI service. Streamlit Cloud handles only the UI and calls the API for predictions — it never loads the model directly, keeping deploys fast and lightweight.

### Model

- **Algorithm**: Calibrated XGBoost (CalibratedClassifierCV with sigmoid calibration)
- **Training data**: 3 years of violent crime records (BATTERY, ASSAULT, ROBBERY) from the Chicago SODA API
- **Spatial grid**: H3 resolution-8 hexagonal tiles (~0.7 km²)
- **Features**: Temporal lags, EWMA momentum, city-normalised rolling stats, spatial neighbour spillover, cyclical time encoding
- **Inference**: Scores all tiles for any date × shift combination

### API Endpoints on Render

   ```toml
   CRIME_API_URL = "https://team23-it5006-chicago-crime-api.onrender.com"
   ```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metadata` | GET | Model config, ROC-AUC, optimal threshold, precision/recall |
| `/baselines` | GET | H3 tile addresses for beat/community mapping |
| `/predict` | POST | Score all tiles for a given date, shift, and threshold |
| `/pr_at_threshold` | GET | Interpolated precision/recall for any threshold value |
| `/docs` | GET | Interactive Swagger UI |

## How to run

### 1. (Re)Train the model

Run `Retrain_Inference_Engine_UI.ipynb` STEP 0 in `ML/App/`. This fetches last 3 years of data from the Chicago SODA API, engineers features, trains the model, and saves deployment artefacts to `ML/Deploy_Render/deployment/`:

- `xgb_calibrated_pipeline.joblib` — calibrated XGBoost pipeline
- `tile_baseline.csv` — per-tile feature baselines (~848 tiles)
- `metadata.json` — model config, performance metrics, and PR curve data

### 2. Update API on Render

1. Push the updated `ML/Deploy_Render/deployment/` folder to GitHub
2. Render auto-redeploys with the new model — Streamlit needs no changes

### 3. View the dashboard on Streamlit Cloud

1. Go to https://team23it5006predictivepolicingay2526sem2-sxfonxxmudo9cyzjct2au.streamlit.app/