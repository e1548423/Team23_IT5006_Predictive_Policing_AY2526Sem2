import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="IT5003 Group23",
        options=["Project Data Overview","Visualization","Summary"],
        icons=["house","bar-chart-line-fill","file-earmark-bar-graph-fill"],
        menu_icon="cast",
        default_index=0
    )

@st.cache_data
def get_kaggle_crime_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "ProjectData", "ChicagoCrimes(20152025).parquet")
    df = pd.read_parquet(file_path)

    # Parsing date column into proper date format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df = df.rename(columns={"Date":"Datetime"})
    df['Date'] = df['Datetime'].dt.date
    df['Time'] = df['Datetime'].dt.time

    # Convert specific object, int64, and float64 columns to category
    cols_to_convert = ["IUCR", "Primary Type", "Description", "Beat", "District", "Ward", "Community Area", "FBI Code"]
    df[cols_to_convert] = df[cols_to_convert].astype("category")
    return df

@st.cache_data
def get_crime_data_clean(df_input):
    df_clean = df_input.dropna()
    df_clean = df_clean[df_clean["Year"] != 2026]
    df_clean["Date"] = df_clean["Datetime"].dt.date
    df_clean["Week"] = df_clean["Datetime"].dt.to_period("W").dt.start_time
    df_clean["Month"] = df_clean["Datetime"].dt.to_period("M").dt.start_time
    df_clean["Year"] = df_clean["Datetime"].dt.year

    return df_clean

df = get_kaggle_crime_data()
df_clean = get_crime_data_clean(df)


# ========== HOME PAGE ==========
if selected == "Project Data Overview":
    st.title("Chicago Crime Dataset - Exploratory Data Analysis")

    st.write("This streamlit application's main objective is to conduct an Exploratory Data Analysis (EDA) on crimes that are occuring "
    "in Chicago from from 2015 - 2015. The main data is obtained from the open source data provided by Chicago Data Portal.")
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
    # ===Dataset Overview===
    st.header("1. Dataset Overview")
    
    # Show column names and data types
    st.subheader("Column Names and Data Types")
    st.table(pd.DataFrame({"Column Name": df.columns, "Data Type": df.dtypes.astype(str).values}))

    # ===Data Preview with Date Range Filter===
    st.header("Data Preview with Date Range")

    # Assume Date column is already datetime in cached parquet
    min_date, max_date = df["Date"].min(), df["Date"].max()

    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    filtered_range_df = df[
        (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
    ]

    st.write(f"Showing {len(filtered_range_df)} rows between {date_range[0]} and {date_range[1]}:")
    st.dataframe(filtered_range_df.head(50))

    # ===Missing Values===
    st.header("Missing Values by Year")

    st.write("The following table displays the number of missing values for each column, grouped by year, to understand the trend of missing data " \
    "over different years. It is discovered that the amount of missing data is small compared to the whole dataset which will not distort the overall EDA when removed. " \
    "Also, this simplifies workflow - avoid complexity of imputing values which can introduce bias if not done carefully.")

    if "Date" in df.columns:
        missing_by_year = (
            df.assign(Year=df["Datetime"].dt.year)
            .groupby("Year")
            .apply(lambda x: x.isnull().sum())
        )
        #st.write(missing_by_year)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(missing_by_year.T, cmap="Reds", annot=True, fmt="d", ax=ax)
        ax.set_title("Missing Values by Year")
        st.pyplot(fig)

# ========== VISUALIZATION PAGE ==========
elif selected == "Visualization":
    # Module 4: Time Series Analysis
    st.header("4. Time Series Analysis: Cases Over Time")

    if "Date" in df_clean.columns and "Case Number" in df_clean.columns:
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
            df_clean.resample(freq, on="Datetime")["Case Number"]
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
            cases["HoverLabel"] = cases["Datetime"].dt.to_period("Q").astype(str)
            x_hover = cases["HoverLabel"]
        else:
            x_hover = cases["Datetime"].dt.strftime(hover_fmt)

        fig.add_trace(go.Scatter(
            x=cases["Datetime"], 
            y=cases["Case Number"],
            mode="lines+markers",
            name=f"Cases by {freq_option}",
            customdata=x_hover,
            hovertemplate="Period: %{customdata}<br>Cases: %{y}<extra></extra>"
        ))

        # 3. Add alternating shaded bands (Now aligned because both use Datetime)
        years = df_clean["Datetime"].dt.year.unique()
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

# ========== SUMMARY PAGE ==========
else:
    st.title(f"Summary page is still empty 🙂, be patient")