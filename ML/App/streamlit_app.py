# =============================================================================
# CHICAGO VIOLENT CRIME PREDICTION — STREAMLIT PATROL DISPATCH DASHBOARD
#
# API-backed version: calls FastAPI on Render for predictions.
# Model is NOT loaded locally — only map/UI logic runs in Streamlit.
#
# Pre-requisites:
#   - Set CRIME_API_URL in Streamlit Cloud secrets or env var
#     e.g. https://chicago-crime-api.onrender.com
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import h3
from datetime import date, datetime, timedelta
from streamlit_folium import st_folium
import folium
import folium.plugins as plugins
from shapely.geometry import shape as shapely_shape, Point

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chicago Crime Prediction — Patrol Dispatch",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API config ───────────────────────────────────────────────────────────────
# Priority: Streamlit secrets > env var > default
API_BASE = (
    st.secrets.get("CRIME_API_URL", None)
    or os.environ.get("CRIME_API_URL", "https://chicago-crime-api.onrender.com")
)

API_LIVE = "https://data.cityofchicago.org/resource/f6bk-yv3r.json"
API_BEATS = "https://data.cityofchicago.org/resource/n9it-hstw.json?$limit=5000"
API_COMMUNITY = "https://data.cityofchicago.org/resource/igwz-8jzy.json?$limit=100"
H3_RES = 8


# =============================================================================
# CACHED LOADERS
# =============================================================================
@st.cache_data(ttl=3600)
def load_metadata():
    """Fetch model metadata from the API."""
    resp = requests.get(f"{API_BASE}/metadata", timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def load_baselines():
    """Fetch tile H3 addresses from the API."""
    resp = requests.get(f"{API_BASE}/baselines", timeout=30)
    resp.raise_for_status()
    return resp.json()["h3_addresses"]


@st.cache_data(ttl=3600)
def load_beats_json():
    """Load beat boundaries as raw JSON list (no geopandas)."""
    resp = requests.get(API_BEATS, timeout=60)
    resp.raise_for_status()
    beats = resp.json()
    return [b for b in beats if "the_geom" in b and isinstance(b["the_geom"], dict)]


@st.cache_data(ttl=3600)
def load_community_json():
    """Load community area boundaries as raw JSON list (no geopandas)."""
    resp = requests.get(API_COMMUNITY, timeout=60)
    resp.raise_for_status()
    areas = resp.json()
    return [a for a in areas if "the_geom" in a and isinstance(a["the_geom"], dict)]


@st.cache_data(ttl=3600)
def build_tile_beat_map(_h3_addresses, _beats_json):
    """Map each H3 tile centroid to a beat/district using point-in-polygon."""
    beat_polys = []
    for b in _beats_json:
        try:
            geom = shapely_shape(b["the_geom"])
            beat_num = b.get("beat_num", b.get("beat", "Unknown"))
            district = b.get("district", "")
            beat_polys.append((geom, str(beat_num), str(district)))
        except Exception:
            continue

    tile_map = {}
    for h3_addr in _h3_addresses:
        lat, lon = h3.cell_to_latlng(h3_addr)
        pt = Point(lon, lat)
        matched = False
        for geom, beat_num, district in beat_polys:
            if geom.contains(pt):
                tile_map[h3_addr] = (beat_num, district)
                matched = True
                break
        if not matched:
            tile_map[h3_addr] = ("Unknown", "")

    return tile_map


@st.cache_data(ttl=3600)
def build_tile_community_map(_h3_addresses, _community_json):
    """Map each H3 tile centroid to a community area name using point-in-polygon."""
    comm_polys = []
    for a in _community_json:
        try:
            geom = shapely_shape(a["the_geom"])
            name = str(a.get("community", a.get("COMMUNITY", "Unknown"))).title()
            comm_polys.append((geom, name))
        except Exception:
            continue

    tile_map = {}
    for h3_addr in _h3_addresses:
        lat, lon = h3.cell_to_latlng(h3_addr)
        pt = Point(lon, lat)
        matched = False
        for geom, name in comm_polys:
            if geom.contains(pt):
                tile_map[h3_addr] = name
                matched = True
                break
        if not matched:
            tile_map[h3_addr] = "Unknown"

    return tile_map


def get_pr_at_threshold(threshold):
    """Fetch interpolated precision/recall for a given threshold from the API."""
    try:
        resp = requests.get(
            f"{API_BASE}/pr_at_threshold",
            params={"threshold": threshold},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_live_lag(target_date):
    """Fetch recent violent crimes for fresh lag_1d."""
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
            lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_RES), axis=1
        )
        yest_df = df[df["Date"].dt.date == yesterday]
        lag_counts = yest_df.groupby("h3_address").size().reset_index(name="lag_1d")
        return lag_counts
    except Exception:
        return None


# =============================================================================
# PREDICTION (via API)
# =============================================================================
def predict_tiles(meta, query_date, shift, threshold, override_tiles=None):
    """Call the FastAPI backend for predictions."""
    payload = {
        "query_date": str(query_date),
        "shift": shift,
        "threshold": threshold,
    }

    # Convert live lag DataFrame to dict for the API
    if override_tiles is not None and len(override_tiles) > 0:
        payload["live_lag"] = dict(
            zip(override_tiles["h3_address"], override_tiles["lag_1d"].astype(float))
        )

    resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    results = pd.DataFrame(data["results"])
    return results.sort_values("crime_probability", ascending=False).reset_index(drop=True)


# =============================================================================
# MAP BUILDER (pure Folium + JSON, no geopandas)
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
    results, beats_json, community_json, tile_beat_map, tile_community_map,
    threshold, show_monitor, district_filter, beat_filter, tier_filter,
):
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="CartoDB positron")

    # ── Beat boundaries ───────────────────────────────────────────────────
    beats_layer = folium.FeatureGroup(name="Police Beats", show=True)
    for b in beats_json:
        beat_num = str(b.get("beat_num", b.get("beat", "Unknown")))
        district = str(b.get("district", ""))
        geojson = b["the_geom"]

        is_sel = (
            (district_filter == "ALL" or district == district_filter)
            and (beat_filter == "ALL" or beat_num == beat_filter)
        )
        style = {
            "fillColor": "#1A237E" if is_sel and district_filter != "ALL" else "transparent",
            "color": "#1A237E",
            "weight": 2.5 if is_sel and district_filter != "ALL" else 1.2,
            "fillOpacity": 0.06 if is_sel and district_filter != "ALL" else 0,
            "dashArray": "" if is_sel and district_filter != "ALL" else "4 4",
        }
        folium.GeoJson(
            data=geojson,
            style_function=lambda feat, s=style: s,
            tooltip=folium.Tooltip(f"<b>Beat {beat_num}</b><br>District {district}"),
        ).add_to(beats_layer)
    beats_layer.add_to(m)

    # ── Community areas ───────────────────────────────────────────────────
    if community_json:
        comm_layer = folium.FeatureGroup(name="Community Areas", show=False)
        for a in community_json:
            name = a.get("community", a.get("COMMUNITY", "Unknown"))
            num = a.get("area_numbe", a.get("AREA_NUMBE", a.get("area_num_1", "")))
            geojson = a["the_geom"]

            folium.GeoJson(
                data=geojson,
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
        community = tile_community_map.get(h3_addr, "Unknown")

        tooltip = (
            f"<div style='font-family:Arial;font-size:13px;'>"
            f"<b>🚨 DISPATCH</b><br>"
            f"<b>Tile:</b> {h3_addr}<br>"
            f"<b>Beat:</b> {b}" + (f" (Dist {d})" if d else "") + "<br>"
            f"<b>Community:</b> {community}<br>"
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
try:
    meta = load_metadata()
    h3_addresses = load_baselines()
except Exception as e:
    st.error(
        f"**Cannot reach the prediction API** at `{API_BASE}`.\n\n"
        f"The Render backend may be cold-starting (takes ~30s on free tier). "
        f"Refresh the page in a moment.\n\n`{e}`"
    )
    st.stop()

beats_json = load_beats_json()
community_json = load_community_json()
tile_beat_map = build_tile_beat_map(tuple(h3_addresses), beats_json)
tile_community_map = build_tile_community_map(tuple(h3_addresses), community_json)

all_districts = sorted({str(b.get("district", "")) for b in beats_json if b.get("district")})
all_beats = sorted({
    str(b.get("beat_num", b.get("beat", "")))
    for b in beats_json
    if b.get("beat_num") or b.get("beat")
})


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
            "morning_noon": "Morning / Noon (06–13)",
            "afternoon_night": "Afternoon / Night (14–21)",
            "overnight": "Overnight (22–05)",
        }[s],
        index=1,
    )
    threshold = st.slider(
        "**Dispatch Threshold**",
        min_value=0.05, max_value=0.50,
        value=float(meta["threshold"]),
        step=0.01, format="%.2f",
        help=(
            "Minimum predicted probability for a tile to be flagged for dispatch. "
            "Raise it to focus on fewer, higher-confidence tiles. "
            "Lower it to cast a wider net — but expect more false positives."
        ),
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
        beat_options = ["ALL"] + sorted({
            str(b.get("beat_num", b.get("beat", "")))
            for b in beats_json
            if str(b.get("district", "")) == district_filter
        })
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
    # Dynamic precision/recall based on current threshold
    pr_data = get_pr_at_threshold(threshold)
    if pr_data:
        dyn_prec = f"{pr_data['precision']:.4f}"
        dyn_rec = f"{pr_data['recall']:.4f}"
    else:
        dyn_prec = meta.get('precision', 'N/A')
        dyn_rec = meta.get('recall', 'N/A')

    st.caption(
        f"**Model:** ROC-AUC {meta['roc_auc']}  \n"
        f"**Precision @ {threshold:.2f}:** {dyn_prec}  \n"
        f"**Recall @ {threshold:.2f}:** {dyn_rec}  \n"
        f"**Threshold:** {meta['threshold']}  \n"
        f"**Tiles:** {meta['tile_count']:,}  \n"
        f"**Trained:** {meta.get('trained_at', 'N/A')[:10]}  \n"
        f"**API:** `{API_BASE}`"
    )


# =============================================================================
# MAIN
# =============================================================================
st.markdown(
    "<h1 style='margin-bottom:0;'>🚨 Chicago Violent Crime Prediction</h1>"
    "<p style='color:#666;font-size:1.1rem;margin-top:0;'>"
    "XGBoost patrol dispatch dashboard — score H3 tiles, flag high-risk areas</p>",
    unsafe_allow_html=True,
)

# ── Prediction ────────────────────────────────────────────────────────────
override = None
if use_live:
    with st.spinner("Fetching live crime data …"):
        override = fetch_live_lag(query_date)
    if override is not None:
        st.info(f"🔄 Live lag: {len(override)} tiles with fresh data from {query_date - timedelta(days=1)}")
    else:
        st.warning("No recent violent crimes in API — using baseline lag values.")

with st.spinner("Running prediction …"):
    results = predict_tiles(meta, query_date, shift, threshold, override)

# ── Metrics ───────────────────────────────────────────────────────────────
n_total = len(results)
n_flagged = int(results["flagged"].sum())
n_critical = int((results["risk_tier"] == "Critical").sum())
n_high = int((results["risk_tier"] == "High").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tiles", f"{n_total:,}")
c2.metric("Flagged", f"{n_flagged:,}", delta=f"{n_flagged/max(n_total,1):.1%}")
c3.metric("Critical", f"{n_critical:,}")
c4.metric("High", f"{n_high:,}")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_map, tab_table, tab_charts = st.tabs(["🗺️ Patrol Map", "📋 Dispatch Table", "📊 Analytics"])

with tab_map:
    with st.spinner("Building map …"):
        patrol_map, filtered_df = build_map(
            results, beats_json, community_json, tile_beat_map, tile_community_map,
            threshold, show_monitor, district_filter, beat_filter, tier_filter,
        )
    st_folium(patrol_map, width=None, height=650, returned_objects=[])

    map_html = patrol_map._repr_html_()
    st.download_button(
        "📥 Download Map (HTML)", data=map_html,
        file_name=f"patrol_map_{query_date}_{shift}.html", mime="text/html",
    )

with tab_table:
    display_df = results.copy()
    display_df["beat"] = display_df["h3_address"].map(lambda h: tile_beat_map.get(h, ("?", ""))[0])
    display_df["district"] = display_df["h3_address"].map(lambda h: tile_beat_map.get(h, ("?", ""))[1])
    display_df["community"] = display_df["h3_address"].map(lambda h: tile_community_map.get(h, "Unknown"))
    if district_filter != "ALL":
        display_df = display_df[display_df["district"] == district_filter]
    if beat_filter != "ALL":
        display_df = display_df[display_df["beat"] == beat_filter]
    if tier_filter:
        display_df = display_df[display_df["risk_tier"].isin(tier_filter)]

    top_n = st.slider("Top N tiles", min_value=5, max_value=50, value=20, step=5)
    show_df = display_df.head(top_n)[
        ["h3_address", "beat", "district", "community", "crime_probability", "risk_tier", "flagged"]
    ].copy()
    show_df["flagged"] = show_df["flagged"].map({1: "🚨 DISPATCH", 0: "monitor"})
    show_df.index = range(1, len(show_df) + 1)
    show_df.index.name = "Rank"

    st.dataframe(
        show_df.style.background_gradient(subset=["crime_probability"], cmap="YlOrRd"),
        use_container_width=True,
    )
    csv = display_df[display_df["flagged"] == 1].to_csv(index=False)
    st.download_button(
        "📥 Download Flagged Tiles (CSV)", data=csv,
        file_name=f"patrol_briefing_{query_date}_{shift}.csv", mime="text/csv",
    )

with tab_charts:
    import matplotlib.pyplot as plt

    tier_palette = {"Critical": "#C0392B", "High": "#E67E22", "Moderate": "#2980B9", "Low": "#27AE60"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.4)

    tier_counts = results["risk_tier"].value_counts().reindex(
        ["Critical", "High", "Moderate", "Low"], fill_value=0
    )
    bars = axes[0].bar(
        tier_counts.index, tier_counts.values,
        color=[tier_palette[t] for t in tier_counts.index],
        alpha=0.85, edgecolor="white", width=0.6,
    )
    for bar, val in zip(bars, tier_counts.values):
        pct = val / max(n_total, 1) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + n_total * 0.005,
            f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold",
        )
    axes[0].set_ylim(0, max(tier_counts.max() * 1.25, 1))
    axes[0].set_ylabel("Number of Tiles")
    axes[0].set_title(f"Risk Tier Distribution — {n_total:,} Tiles\n"
                      f"{query_date} | {shift.replace('_',' ').title()}", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(results["crime_probability"], bins=40, color="#2980B9", alpha=0.8, edgecolor="white")
    axes[1].axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.2f})")
    axes[1].set_xlabel("Crime Probability")
    axes[1].set_ylabel("Number of Tiles")
    axes[1].set_title("Probability Distribution", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    st.pyplot(fig)
