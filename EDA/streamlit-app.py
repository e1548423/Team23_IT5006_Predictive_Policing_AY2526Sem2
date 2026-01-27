import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURATION ---
FILE_ID = "1sf2LEsakAjMzEqayzZ-1maf0aYXHYm1a"
CSV_FILE = "project_data.csv"
PARQUET_FILE = "project_data.parquet"

@st.cache_data
def get_data_from_gdrive(file_id):
    """Download CSV from Google Drive, convert to Parquet, and return dataframe"""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, CSV_FILE, quiet=False)

    # Load CSV
    df = pd.read_csv(CSV_FILE, on_bad_lines="skip")

    # Ensure Date column is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        st.error("No 'Date' column found in dataset for time series analysis.")

    # Convert specific object, int64, and float64 columns to category
    cols_to_convert = ["IUCR", "Primary Type", "Description", "Beat", "District", "Ward", "Community Area", "FBI Code"]
    df[cols_to_convert] = df[cols_to_convert].astype("category")

    # Convert and cache as Parquet
    df.to_parquet(PARQUET_FILE, index=False)
    return df

# --- CACHE LOGIC ---
st.subheader("Data Source Status")

if os.path.exists(PARQUET_FILE):
    # Load from local Parquet cache
    df = pd.read_parquet(PARQUET_FILE)
    st.success(f"🚀 Loaded {len(df):,} rows from local cache!")
    if st.button("🔄 Refresh from Google Drive"):
        os.remove(PARQUET_FILE)
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
        st.rerun()
else:
    # Download and convert one time only
    with st.spinner("Downloading dataset from Google Drive... this may take a minute."):
        df = get_data_from_gdrive(FILE_ID)
        st.success("✅ Downloaded CSV, converted to Parquet, and cached locally!")

try:
    st.title("Chicago Crime Dataset - Exploratory Data Analysis")

    # Module 1: Dataset Overview
    st.header("1. Dataset Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Show column names and data types
    st.subheader("Column Names and Data Types")
    st.table(pd.DataFrame({"Column Name": df.columns, "Data Type": df.dtypes.values}))

    # Module 2: Data Preview with Date Range Filter
    st.header("2. Data Preview with Date Range")

    # Assume Date column is already datetime in cached parquet
    min_date, max_date = df["Date"].min(), df["Date"].max()

    date_range = st.slider(
        "Select date range",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date())
    )

    filtered_range_df = df[
        (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
    ]

    st.write(f"Showing {len(filtered_range_df)} rows between {date_range[0]} and {date_range[1]}:")
    st.dataframe(filtered_range_df.head(50))

    # Module 3: Missing Values 
    st.header("3. Missing Values by Year")
    if "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
        missing_by_year = df.groupby("Year").apply(lambda x: x.isnull().sum())
        #st.write(missing_by_year)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(missing_by_year.T, cmap="Reds", annot=True, fmt="d", ax=ax)
        ax.set_title("Missing Values by Year")
        st.pyplot(fig)

    # Module 4: Time Series Analysis
    st.header("4. Time Series Analysis: Cases Over Time")

    if "Date" in df.columns and "Case Number" in df.columns:
        # Frequency selector
        freq_option = st.selectbox(
            "Select frequency for time series",
            options=["Year", "Quarter", "Month"],
            index=0
        )

        freq_map = {"Year": "YS", "Quarter": "QS", "Month": "MS"} # Use 'S' for Start of period
        freq = freq_map[freq_option]

        # 1. Resample - Keep 'Date' as a datetime column
        cases = (
            df.resample(freq, on="Date")["Case Number"]
            .count()
            .reset_index()
        )

        # Drop empty bins
        cases = cases[cases["Case Number"] > 0]

        # 2. Create figure - Use the actual DATE column for X
        fig = go.Figure()
        
        # Determine tooltip format based on selection
        hover_fmt = "%Y" if freq_option == "Year" else "%b %Y"
        if freq_option == "Quarter":
            # Plotly doesn't have a native 'Q' hover format, so we'll pre-format a custom hover column
            cases["HoverLabel"] = cases["Date"].dt.to_period("Q").astype(str)
            x_hover = cases["HoverLabel"]
        else:
            x_hover = cases["Date"].dt.strftime(hover_fmt)

        fig.add_trace(go.Scatter(
            x=cases["Date"], 
            y=cases["Case Number"],
            mode="lines+markers",
            name=f"Cases by {freq_option}",
            customdata=x_hover,
            hovertemplate="Period: %{customdata}<br>Cases: %{y}<extra></extra>"
        ))

        # 3. Add alternating shaded bands (Now aligned because both use Datetime)
        years = df["Date"].dt.year.unique()
        for i, year in enumerate(sorted(years)):
            fig.add_vrect(
                x0=f"{year}-01-01", x1=f"{year}-12-31",
                fillcolor="lightgrey" if i % 2 == 0 else "white",
                opacity=0.2,
                layer="below",
                line_width=0
            )

        # 4. Update layout with proper axis formatting
        dt_format = "%Y"
        if freq_option == "Month":
            dt_format = "%b %Y"
        elif freq_option == "Quarter":
            # Plotly trick to show Quarters on axis
            fig.update_xaxes(tickformat="Q%q\n%Y") 

        fig.update_layout(
            title=f"Cases by {freq_option}",
            xaxis_title=freq_option,
            yaxis_title="Number of Cases",
            hovermode="x unified",
            xaxis=dict(tickformat=dt_format if freq_option != "Quarter" else None)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No 'Date' or 'Case Number' column found.")

except Exception as e:
    st.error(f"Error loading file: {e}")
