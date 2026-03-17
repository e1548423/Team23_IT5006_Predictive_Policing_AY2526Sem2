# =============================================================================
# CHICAGO VIOLENT CRIME PREDICTION — STREAMLIT PATROL DISPATCH DASHBOARD
#
# Inference-only app: loads pre-trained XGBoost model + tile baselines,
# scores tiles, renders interactive Folium map with beat/community overlays.
#
# Pre-requisites in deployment/ folder (committed to your GitHub repo):
#   - xgb_calibrated_pipeline.joblib
#   - tile_baseline.csv
#   - metadata.json
#
# Run locally:  streamlit run streamlit_app.py
# Deploy:       push to GitHub → connect to Streamlit Cloud
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import requests
import h3
import geopandas as gpd
from datetime import date, datetime, timedelta
from shapely.geometry import shape
from streamlit_folium import st_folium
import folium
import folium.plugins as plugins

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chicago Crime Prediction — Patrol Dispatch",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "..", "deployment")
API_LIVE = "https://data.cityofchicago.org/resource/f6bk-yv3r.json"
H3_RES = 8
VIOLENT_TYPES = ["BATTERY", "ASSAULT", "ROBBERY"]


# =============================================================================
# CACHED LOADERS — run once, persist across reruns
# =============================================================================
@st.cache_resource
def load_model():
    """Load calibrated XGBoost pipeline + metadata."""
    pipeline = joblib.load(os.path.join(DEPLOY_DIR, "xgb_calibrated_pipeline.joblib"))
    baselines = pd.read_csv(os.path.join(DEPLOY_DIR, "tile_baseline.csv"))
    with open(os.path.join(DEPLOY_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return pipeline, baselines, meta


@st.cache_data(ttl=3600)
def load_beats():
    """Load Chicago police beat boundaries from SODA API."""
    url = "https://data.cityofchicago.org/resource/n9it-hstw.json?$limit=5000"
    raw = pd.read_json(url)
    if "the_geom" in raw.columns:
        gdf = gpd.GeoDataFrame(
            raw.drop(columns=["the_geom"]).copy(),
            geometry=raw["the_geom"].apply(
                lambda g: shape(g) if isinstance(g, dict) else None
            ),
            crs="EPSG:4326",
        )
        gdf = gdf[gdf.geometry.notna()].copy()
    else:
        gdf = gpd.GeoDataFrame(raw.copy(), geometry=None, crs="EPSG:4326")
    return gdf


@st.cache_data(ttl=3600)
def load_community_areas():
    """Load Chicago community area boundaries from SODA GeoJSON."""
    url = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"
    try:
        gdf = gpd.read_file(url)
        gdf = gdf[gdf.geometry.notna()].copy()
        # Normalise column names
        renames = {}
        for orig, target in [
            ("COMMUNITY", "community"), ("name", "community"),
            ("AREA_NUMBE", "area_numbe"), ("area_num_1", "area_numbe"),
        ]:
            if orig in gdf.columns and target not in gdf.columns:
                renames[orig] = target
        if renames:
            gdf = gdf.rename(columns=renames)
        return gdf
    except Exception:
        return None


@st.cache_data(ttl=3600)
def build_tile_beat_map(baselines, beats_gdf):
    """Spatial join: map each H3 tile to a police beat/district."""
    tile_h3 = baselines[["h3_address"]].drop_duplicates().copy()
    tile_h3[["_lat", "_lon"]] = tile_h3["h3_address"].apply(
        lambda x: pd.Series(h3.cell_to_latlng(x))
    )
    tile_pts = gpd.GeoDataFrame(
        tile_h3,
        geometry=gpd.points_from_xy(tile_h3["_lon"], tile_h3["_lat"]),
        crs="EPSG:4326",
    )

    bcol = (
        "beat_num"
        if "beat_num" in beats_gdf.columns
        else ("beat" if "beat" in beats_gdf.columns else None)
    )
    dcol = "district" if "district" in beats_gdf.columns else None
    join_cols = ["geometry"] + ([bcol] if bcol else []) + ([dcol] if dcol else [])

    joined = gpd.sjoin(tile_pts, beats_gdf[join_cols], how="left", predicate="within")

    tile_map = {}
    for _, r in joined.iterrows():
        b = str(r.get(bcol, "Unknown")) if bcol and pd.notna(r.get(bcol)) else "Unknown"
        d = str(r.get(dcol, "")) if dcol and pd.notna(r.get(dcol)) else ""
        tile_map[r["h3_address"]] = (b, d)

    return tile_map


@st.cache_data(ttl=300)
def fetch_live_lag(target_date):
    """Fetch recent violent crimes to compute fresh lag_1d per tile."""
    yesterday = target_date - timedelta(days=1)
    start = target_date - timedelta(days=2)
    params = {
        "$where": (
            f"date >= '{start.isoformat()}' "
            f"AND date < '{(target_date + timedelta(days=1)).isoformat()}' "
            f"AND primary_type in('BATTERY','ASSAULT','ROBBERY')"
        ),
        "$limit": 50000,
        "$order": "date ASC",
    }
    try:
        resp = requests.get(API_LIVE, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
        df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
        df = df.dropna(subset=["Date", "latitude", "longitude"])
        df["h3_address"] = df.apply(
            lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_RES),
            axis=1,
        )
        yest_df = df[df["Date"].dt.date == yesterday]
        lag_counts = yest_df.groupby("h3_address").size().reset_index(name="lag_1d")
        return lag_counts
    except Exception:
        return None


# =============================================================================
# PREDICTION ENGINE
# =============================================================================
SHIFT_MAP = {
    "morning_noon": {"is_afternoon_night": 0, "is_overnight": 0, "hour_start": 6},
    "afternoon_night": {"is_afternoon_night": 1, "is_overnight": 0, "hour_start": 14},
    "overnight": {"is_afternoon_night": 0, "is_overnight": 1, "hour_start": 22},
}


def predict_tiles(pipeline, baselines, meta, query_date, shift, threshold, override_tiles=None):
    """Score all tiles for a given date and shift."""
    tiles = baselines.copy()
    feature_cols = meta["feature_cols"]

    # Apply live lag overrides
    if override_tiles is not None and len(override_tiles) > 0:
        tiles = tiles.merge(
            override_tiles[["h3_address", "lag_1d"]].rename(
                columns={"lag_1d": "lag_1d_fresh"}
            ),
            on="h3_address",
            how="left",
        )
        tiles["lag_1d"] = tiles["lag_1d_fresh"].fillna(tiles["lag_1d"])
        tiles.drop(columns=["lag_1d_fresh"], inplace=True)

    # Shift flags
    s = SHIFT_MAP[shift]
    tiles["is_afternoon_night"] = s["is_afternoon_night"]
    tiles["is_overnight"] = s["is_overnight"]

    # Cyclical time features — (month-1)/12 matches training notebook
    dt = pd.Timestamp(query_date)
    dow, mon = dt.dayofweek, dt.month
    tiles["day_sin"] = np.sin(2 * np.pi * dow / 7)
    tiles["day_cos"] = np.cos(2 * np.pi * dow / 7)
    tiles["month_sin"] = np.sin(2 * np.pi * (mon - 1) / 12)
    tiles["month_cos"] = np.cos(2 * np.pi * (mon - 1) / 12)

    # Score
    X = tiles[feature_cols]
    probs = pipeline.predict_proba(X)[:, 1]

    results = tiles[["h3_address"]].copy()
    results["crime_probability"] = probs.round(4)
    results["flagged"] = (probs >= threshold).astype(int)
    results["risk_tier"] = pd.cut(
        probs,
        bins=[0, 0.10, 0.20, 0.35, 1.0],
        labels=["Low", "Moderate", "High", "Critical"],
    )
    results["shift"] = shift
    results["query_date"] = str(query_date)
    return results.sort_values("crime_probability", ascending=False).reset_index(drop=True)


# =============================================================================
# MAP BUILDER
# =============================================================================
def prob_to_hex(prob, threshold=0.15):
    if prob < threshold:
        return "#A8D5A2"
    t = min((prob - threshold) / (0.40 - threshold), 1.0)
    r = int(231 * t + 39 * (1 - t))
    g = int(76 * t + 174 * (1 - t))
    b = int(60 * t + 96 * (1 - t))
    return f"#{r:02X}{g:02X}{b:02X}"


def build_map(
    results, beats_gdf, community_gdf, tile_beat_map,
    threshold, show_monitor, district_filter, beat_filter, tier_filter,
):
    m = folium.Map(
        location=[41.8781, -87.6298],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # ── Beat boundaries ───────────────────────────────────────────────────
    bcol = "beat_num" if "beat_num" in beats_gdf.columns else ("beat" if "beat" in beats_gdf.columns else None)
    dcol = "district" if "district" in beats_gdf.columns else None
    beats_layer = folium.FeatureGroup(name="Police Beats", show=True)
    beats_4326 = beats_gdf.to_crs("EPSG:4326")

    for _, row in beats_4326.iterrows():
        bnum = row.get(bcol, "Unknown") if bcol else "Unknown"
        dist = row.get(dcol, "") if dcol else ""
        is_sel = (district_filter == "ALL" or str(dist) == district_filter) and (
            beat_filter == "ALL" or str(bnum) == beat_filter
        )
        style = {
            "fillColor": "#1A237E" if is_sel and district_filter != "ALL" else "transparent",
            "color": "#1A237E",
            "weight": 2.5 if is_sel and district_filter != "ALL" else 1.2,
            "fillOpacity": 0.06 if is_sel and district_filter != "ALL" else 0,
            "dashArray": "" if is_sel and district_filter != "ALL" else "4 4",
        }
        folium.GeoJson(
            data=row.geometry.__geo_interface__,
            style_function=lambda feat, s=style: s,
            tooltip=folium.Tooltip(f"<b>Beat {bnum}</b><br>District {dist}"),
        ).add_to(beats_layer)
    beats_layer.add_to(m)

    # ── Community areas ───────────────────────────────────────────────────
    if community_gdf is not None and len(community_gdf) > 0:
        comm_layer = folium.FeatureGroup(name="Community Areas", show=False)
        comm_4326 = community_gdf.to_crs("EPSG:4326")
        for _, crow in comm_4326.iterrows():
            name = crow.get("community", "Unknown")
            num = crow.get("area_numbe", "")
            geom = crow.geometry
            if geom is None:
                continue
            folium.GeoJson(
                data=geom.__geo_interface__,
                style_function=lambda feat: {
                    "fillColor": "transparent",
                    "color": "#6A1B9A",
                    "weight": 2.0,
                    "fillOpacity": 0,
                    "dashArray": "6 3",
                },
                tooltip=folium.Tooltip(f"<b>{str(name).title()}</b><br>Area {num}"),
            ).add_to(comm_layer)
        comm_layer.add_to(m)

    # ── Filter results ────────────────────────────────────────────────────
    df = results.copy()
    df["_beat"] = df["h3_address"].map(lambda h: tile_beat_map.get(h, ("Unknown", ""))[0])
    df["_district"] = df["h3_address"].map(lambda h: tile_beat_map.get(h, ("Unknown", ""))[1])

    if district_filter != "ALL":
        df = df[df["_district"] == district_filter]
    if beat_filter != "ALL":
        df = df[df["_beat"] == beat_filter]
    if tier_filter:
        df = df[df["risk_tier"].isin(tier_filter)]

    # ── Flagged tiles ─────────────────────────────────────────────────────
    flagged_layer = folium.FeatureGroup(name="🚨 Flagged Tiles", show=True)
    flagged_df = df[df["flagged"] == 1]

    for _, row in flagged_df.iterrows():
        prob = float(row["crime_probability"])
        tier = str(row["risk_tier"])
        h3_addr = row["h3_address"]
        colour = prob_to_hex(prob, threshold)
        boundary = h3.cell_to_boundary(h3_addr)
        b, d = tile_beat_map.get(h3_addr, ("?", ""))

        tooltip = (
            f"<div style='font-family:Arial;font-size:13px;'>"
            f"<b>🚨 DISPATCH</b><br>"
            f"<b>Tile:</b> {h3_addr}<br>"
            f"<b>Beat:</b> {b}" + (f" (Dist {d})" if d else "") + "<br>"
            f"<b>Risk:</b> {tier}<br>"
            f"<b>Prob:</b> {prob:.1%}"
            f"</div>"
        )
        folium.Polygon(
            locations=[(lat, lon) for lat, lon in boundary],
            color=colour, weight=1.5, fill=True,
            fill_color=colour, fill_opacity=0.72,
            tooltip=folium.Tooltip(tooltip, sticky=True),
            popup=folium.Popup(tooltip, max_width=280),
        ).add_to(flagged_layer)
    flagged_layer.add_to(m)

    # ── Monitor tiles ─────────────────────────────────────────────────────
    if show_monitor:
        monitor_layer = folium.FeatureGroup(name="⚠️ Monitor", show=True)
        monitor_df = df[(df["flagged"] == 0) & (df["crime_probability"] >= threshold * 0.65)]
        for _, row in monitor_df.iterrows():
            boundary = h3.cell_to_boundary(row["h3_address"])
            folium.Polygon(
                locations=[(lat, lon) for lat, lon in boundary],
                color="#F39C12", weight=1.0, fill=True,
                fill_color="#F39C12", fill_opacity=0.30,
                tooltip=folium.Tooltip(f"Monitor: {float(row['crime_probability']):.1%}"),
            ).add_to(monitor_layer)
        monitor_layer.add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    plugins.MiniMap(tile_layer="CartoDB positron", zoom_level_offset=-6).add_to(m)

    # Auto-zoom to filtered area
    if (district_filter != "ALL" or beat_filter != "ALL") and len(df) > 0:
        lats = df["h3_address"].apply(lambda h: h3.cell_to_latlng(h)[0])
        lons = df["h3_address"].apply(lambda h: h3.cell_to_latlng(h)[1])
        m.fit_bounds([[lats.min(), lons.min()], [lats.max(), lons.max()]])

    return m, df


# =============================================================================
# LOAD DATA
# =============================================================================
pipeline, baselines, meta = load_model()
beats_gdf = load_beats()
community_gdf = load_community_areas()
tile_beat_map = build_tile_beat_map(baselines, beats_gdf)

all_districts = sorted({d for _, d in tile_beat_map.values() if d})
all_beats = sorted({b for b, _ in tile_beat_map.values() if b != "Unknown"})


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    query_date = st.date_input("**Date**", value=date.today())

    shift = st.selectbox(
        "**Shift**",
        options=["morning_noon", "afternoon_night", "overnight"],
        format_func=lambda s: {
            "morning_noon": "Morning / Noon (06:00–13:59)",
            "afternoon_night": "Afternoon / Night (14:00–21:59)",
            "overnight": "Overnight (22:00–05:59)",
        }[s],
        index=1,
    )

    threshold = st.slider(
        "**Dispatch Threshold**",
        min_value=0.05, max_value=0.50,
        value=float(meta["threshold"]),
        step=0.01, format="%.2f",
    )

    st.markdown("---")
    st.markdown("### 📍 Area Filter")

    district_filter = st.selectbox(
        "District",
        options=["ALL"] + all_districts,
        format_func=lambda d: "All districts" if d == "ALL" else f"District {d}",
    )

    if district_filter == "ALL":
        beat_options = ["ALL"] + all_beats
    else:
        beat_options = ["ALL"] + sorted(
            {b for b, d in tile_beat_map.values() if d == district_filter and b != "Unknown"}
        )

    beat_filter = st.selectbox(
        "Beat",
        options=beat_options,
        format_func=lambda b: "All beats" if b == "ALL" else f"Beat {b}",
    )

    st.markdown("---")
    st.markdown("### 🏷️ Risk Tiers")
    tier_filter = st.multiselect(
        "Show tiers",
        options=["Critical", "High", "Moderate", "Low"],
        default=["Critical", "High", "Moderate", "Low"],
    )

    show_monitor = st.checkbox("Show monitor layer", value=False)

    st.markdown("---")
    use_live = st.checkbox("🔄 Use live lag data", value=False)

    st.markdown("---")
    st.caption(
        f"**Model:** ROC-AUC {meta['roc_auc']}  \n"
        f"**Threshold:** {meta['threshold']}  \n"
        f"**Tiles:** {len(baselines):,}  \n"
        f"**Trained:** {meta.get('trained_at', 'N/A')[:10]}"
    )


# =============================================================================
# MAIN AREA
# =============================================================================
st.markdown(
    """
    <h1 style="margin-bottom:0;">🚨 Chicago Violent Crime Prediction</h1>
    <p style="color:#666; font-size:1.1rem; margin-top:0;">
        XGBoost patrol dispatch dashboard — score H3 tiles, flag high-risk areas, generate patrol maps
    </p>
    """,
    unsafe_allow_html=True,
)

# ── Run prediction ────────────────────────────────────────────────────────
override = None
if use_live:
    with st.spinner("Fetching live crime data …"):
        override = fetch_live_lag(query_date)
    if override is not None:
        st.info(f"🔄 Live lag loaded: {len(override)} tiles with fresh data from {query_date - timedelta(days=1)}")
    else:
        st.warning("No recent violent crimes found in API — using baseline lag values.")

results = predict_tiles(pipeline, baselines, meta, query_date, shift, threshold, override)

# ── Summary metrics ───────────────────────────────────────────────────────
n_total = len(results)
n_flagged = int(results["flagged"].sum())
n_critical = int((results["risk_tier"] == "Critical").sum())
n_high = int((results["risk_tier"] == "High").sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tiles", f"{n_total:,}")
col2.metric("Flagged for Dispatch", f"{n_flagged:,}", delta=f"{n_flagged/n_total:.1%}")
col3.metric("Critical Risk", f"{n_critical:,}")
col4.metric("High Risk", f"{n_high:,}")

# ── Tabs: Map / Table / Charts ────────────────────────────────────────────
tab_map, tab_table, tab_charts = st.tabs(["🗺️ Patrol Map", "📋 Dispatch Table", "📊 Analytics"])

with tab_map:
    with st.spinner("Building map …"):
        patrol_map, filtered_df = build_map(
            results, beats_gdf, community_gdf, tile_beat_map,
            threshold, show_monitor, district_filter, beat_filter, tier_filter,
        )
    st_folium(patrol_map, width=None, height=650, returned_objects=[])

    # Download map as HTML
    from io import BytesIO
    map_html = patrol_map._repr_html_()
    st.download_button(
        "📥 Download Map (HTML)",
        data=map_html,
        file_name=f"patrol_map_{query_date}_{shift}.html",
        mime="text/html",
    )

with tab_table:
    # Apply same filters
    display_df = results.copy()
    display_df["beat"] = display_df["h3_address"].map(
        lambda h: tile_beat_map.get(h, ("?", ""))[0]
    )
    display_df["district"] = display_df["h3_address"].map(
        lambda h: tile_beat_map.get(h, ("?", ""))[1]
    )
    if district_filter != "ALL":
        display_df = display_df[display_df["district"] == district_filter]
    if beat_filter != "ALL":
        display_df = display_df[display_df["beat"] == beat_filter]
    if tier_filter:
        display_df = display_df[display_df["risk_tier"].isin(tier_filter)]

    top_n = st.slider("Top N tiles", min_value=5, max_value=50, value=20, step=5)

    show_df = display_df.head(top_n)[
        ["h3_address", "beat", "district", "crime_probability", "risk_tier", "flagged"]
    ].copy()
    show_df["flagged"] = show_df["flagged"].map({1: "🚨 DISPATCH", 0: "monitor"})
    show_df.index = range(1, len(show_df) + 1)
    show_df.index.name = "Rank"

    st.dataframe(
        show_df.style.background_gradient(
            subset=["crime_probability"], cmap="YlOrRd"
        ),
        use_container_width=True,
    )

    # CSV download
    csv = display_df[display_df["flagged"] == 1].to_csv(index=False)
    st.download_button(
        "📥 Download Flagged Tiles (CSV)",
        data=csv,
        file_name=f"patrol_briefing_{query_date}_{shift}.csv",
        mime="text/csv",
    )

with tab_charts:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    tier_palette = {
        "Critical": "#C0392B",
        "High": "#E67E22",
        "Moderate": "#2980B9",
        "Low": "#27AE60",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.4)

    # Panel 1: Risk tier distribution
    tier_counts = results["risk_tier"].value_counts().reindex(
        ["Critical", "High", "Moderate", "Low"], fill_value=0
    )
    bars = axes[0].bar(
        tier_counts.index, tier_counts.values,
        color=[tier_palette[t] for t in tier_counts.index],
        alpha=0.85, edgecolor="white", width=0.6,
    )
    for bar, val in zip(bars, tier_counts.values):
        pct = val / n_total * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + n_total * 0.005,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", fontsize=9, fontweight="bold",
        )
    axes[0].set_ylim(0, tier_counts.max() * 1.25)
    axes[0].set_ylabel("Number of Tiles")
    axes[0].set_title(
        f"Risk Tier Distribution — {n_total:,} Tiles\n"
        f"{query_date} | {shift.replace('_', ' ').title()}",
        fontweight="bold",
    )
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Probability histogram
    axes[1].hist(
        results["crime_probability"], bins=40,
        color="#2980B9", alpha=0.8, edgecolor="white",
    )
    axes[1].axvline(threshold, color="red", linestyle="--", linewidth=2,
                    label=f"Threshold ({threshold:.2f})")
    axes[1].set_xlabel("Crime Probability")
    axes[1].set_ylabel("Number of Tiles")
    axes[1].set_title("Probability Distribution", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    st.pyplot(fig)
