import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import json
import geopandas as gpd
from shapely import wkt
import kagglehub
from kagglehub import KaggleDatasetAdapter

KAGGLE_ACC = 'rkyz801/'
DATASET_ID = 'it5006chicagocrimeparquet'
FILE_LIST = ["ChicagoCrimes(20152025).parquet","ChicagoCommunityArea.parquet"]


@st.cache_data
def get_data_kaggle_crime(file_id):
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_ACC + DATASET_ID,
        file_id,
        )
    return df

if os.path.exists(FILE_LIST[0]):
    # Load from local Parquet cache
    df_crime = pd.read_parquet(FILE_LIST[0])
    st.success(f"{FILE_LIST[0]} Loaded {len(df_crime):,} rows from local cache!")
    if st.button("🔄 Refresh from Kaggle"):
        os.remove(FILE_LIST[0])
        if os.path.exists(FILE_LIST[0]):
            os.remove(FILE_LIST[0])
        st.rerun()
else:
    # Download and convert one time only
    with st.spinner("Downloading dataset from Kaggle... this may take a minute."):
        df_crime = get_data_kaggle_crime(FILE_LIST[0])
        st.success(f"✅ {FILE_LIST[0] }Downloaded , and cached locally!")


if os.path.exists(FILE_LIST[1]):
    # Load from local Parquet cache
    df_polygon = pd.read_parquet(FILE_LIST[1])
    st.success(f"{FILE_LIST[1]} Loaded {len(df_polygon):,} rows from local cache!")
    if st.button("🔄 Refresh from Kaggle"):
        os.remove(FILE_LIST[1])
        if os.path.exists(FILE_LIST[1]):
            os.remove(FILE_LIST[1])
        st.rerun()
else:
    # Download and convert one time only
    with st.spinner("Downloading dataset from Kaggle... this may take a minute."):
        df_polygon = get_data_kaggle_crime(FILE_LIST[1])
        st.success(f"✅ {FILE_LIST[1] }Downloaded, and cached locally!")

@st.cache_data
def process_crime_data(df_main):
    # 1. Standard Date Processing
    df_main['Date_Temp'] = pd.to_datetime(df_main['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df_main = df_main.rename(columns={"Date": "Original_Date_Str", "Date_Temp": "Datetime"})
    
    # Extract temporal features
    df_main['Date'] = df_main['Datetime'].dt.date
    df_main['Time'] = df_main['Datetime'].dt.time
    df_main['Month'] = df_main['Datetime'].dt.month
    df_main['Year'] = df_main['Datetime'].dt.year

    # 2. Handling Duplicates (Keep latest based on 'ID')
    # Sort by Case Number and ID (Descending)
    df_main = df_main.sort_values(by=['Case Number', 'ID'], ascending=[True, False])
    
    # Drop duplicates, keeping the first (which is now the latest due to sort)
    df_main = df_main.drop_duplicates(subset=['Case Number'], keep='first')

    # 3. Handling Missing Data
    # Removes any row where critical data is missing (e.g., Latitude, Longitude, or Ward)
    df_main = df_main.dropna()

    # 4. Memory Optimization: Convert to Categories
    cols_to_convert = ["IUCR", "Primary Type", "Description", "Beat", "District", "Ward", "Community Area", "FBI Code"]
    df_main[cols_to_convert] = df_main[cols_to_convert].astype("category")
    
    return df_main


@st.cache_data
def process_geo_data(df_main):
    df_process = df_main
    df_process.columns = ['GEOMETRY','AREA_NUMBER','COMMUNITY','AREA_NUM_1','SHAPE_AREA','SHAPE_LEN']

    # Convert WKT string to geometry objects and convert area to km2
    df_process["GEOMETRY_OBJ"] = df_process["GEOMETRY"].apply(wkt.loads)
    df_process['SHAPE_AREA_FLT'] = df_process['SHAPE_AREA'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
    df_process['AREA_KM2']  = round(df_process['SHAPE_AREA_FLT'] * 9.2903e-8,2)

    # Create GeoDataFrame
    geo_data = gpd.GeoDataFrame(
        df_process,
        geometry="GEOMETRY_OBJ",
        crs="EPSG:4326"
    )

    return geo_data

@st.cache_data
def get_crime_data_clean(df_input):
    df_clean = df_input.dropna()
    df_clean = df_clean[df_clean["Year"] != 2026]
    df_clean["Date"] = df_clean["Datetime"].dt.date
    df_clean["Week"] = df_clean["Datetime"].dt.isocalendar().week
    df_clean["Month"] = df_clean["Datetime"].dt.month
    df_clean["Year"] = df_clean["Datetime"].dt.year

    return df_clean

@st.cache_data
def process_gdf_yearly(df_main, _geo_main):
    df_crime_yearly = df_main.groupby(['Community Area','Year']).size().reset_index(name='Total Crime')


    gdf_yearly = _geo_main.copy().merge(
        df_crime_yearly,
        left_on="AREA_NUMBER",
        right_on="Community Area",
        how="left"
    )

    gdf_yearly['Crime/km2'] = round(gdf_yearly['Total Crime'] / (gdf_yearly['AREA_KM2']),0)

    return gdf_yearly

@st.cache_data
def process_crimes(df_main):
    grouped_crime = df_main.groupby(['Primary Type','Year','Month']).size().reset_index(name='Crime Count')
    grouped_crime = grouped_crime.sort_values(by=['Primary Type', 'Year', 'Month']).reset_index(drop=True)
    grouped_crime = grouped_crime.loc[grouped_crime['Primary Type'] != 'OTHER OFFENSE']
    grouped_crime['Cummulative Count'] = grouped_crime.groupby(['Primary Type'])['Crime Count'].cumsum()


    top_crime = (
        grouped_crime
        .sort_values(["Year", "Cummulative Count"], ascending=[True, False])
        .loc[grouped_crime['Month']==12]
        .groupby("Year", group_keys=False)
        .head(11)
    )
    top_crime_list = list(top_crime.loc[(top_crime['Year'] == 2025) & (top_crime['Month']==12)]['Primary Type'])

    return top_crime, top_crime_list

def time_day_func(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

def crime_percent_time(df_main,crime_list):

    df_time = df_main[['Primary Type','Date','Time','Year','Month']].reset_index(drop=True)
    df_time['Hour'] = df_time['Time'].apply(lambda x: (x.hour))
    df_time['Time of Day'] = df_time['Hour'].apply(lambda x: time_day_func(x))
    df_time['Top Crime'] = df_time['Primary Type'].apply(lambda x: x in crime_list)

    df_time = df_time.loc[df_time['Top Crime'] == True].reset_index(drop=True)
    df_time_Occurrence = df_time.groupby(['Time of Day','Primary Type']).size().reset_index(name='Occurrence')

    time_of_day_list = ['Morning', 'Afternoon', 'Evening', 'Night']
    total_crime_day = []

    for time in time_of_day_list:
        df_temp = df_time_Occurrence.loc[df_time_Occurrence['Time of Day'] == time].reset_index(drop=True)
        sum_crime = sum(df_temp['Occurrence'])
        total_crime_day.append(sum_crime)

    crime_Occurrence = []

    for crime in crime_list:
        df_temp = df_time_Occurrence.loc[df_time_Occurrence['Primary Type'] == crime].reset_index(drop=True)
        sum_crime = sum(df_temp['Occurrence'])
        crime_Occurrence.append(sum_crime)

    sum_crime_time = pd.DataFrame({'Primary Type':crime_list,'Sum Crime': crime_Occurrence})

    crime_percentage = pd.merge(df_time_Occurrence, sum_crime_time, on='Primary Type', how='inner')
    crime_percentage['CrimePercentage'] = round(crime_percentage['Occurrence']/crime_percentage['Sum Crime'],3)*100

    time_summary = crime_percentage.copy().groupby(['Time of Day'])['Occurrence'].sum().reset_index(name='Crime Count')

    return crime_percentage, time_summary

def crime_group(df_main):
    # Aggregate counts by Year and Month
    df_group = (
        df_main.groupby(["Year", "Month"])
        .size()
        .reset_index(name="Cases")
    )
    return df_group

def crime_cases_group(df_main):
    # Assuming df_clean already has a 'Datetime' column parsed
    df_main = df_main.dropna()
    df_main = df_main[df_main["Year"] != 2026]
    df_main["Date"] = df_main["Datetime"].dt.date
    df_main["Week"] = df_main["Datetime"].dt.to_period("W").dt.start_time
    df_main["Month"] = df_main["Datetime"].dt.month
    df_main["Year"] = df_main["Datetime"].dt.year

    # Aggregate counts using df_clean
    group_date = df_main.groupby("Date").size().reset_index(name="Cases")
    group_week = df_main.groupby("Week").size().reset_index(name="Cases")
    group_month = df_main.groupby("Month").size().reset_index(name="Cases")
    group_year = df_main.groupby("Year").size().reset_index(name="Cases")
    return group_date, group_week, group_month, group_year

with st.sidebar:
    selected = option_menu(
        menu_title="IT5003 Group23",
        options=["Project Data Overview","Visualization","Summary"],
        icons=["house","bar-chart-line-fill","file-earmark-bar-graph-fill"],
        menu_icon="cast",
        default_index=0
    )

# ==========Retrieving and Processing Data==========
df_clean = process_crime_data(df_crime)
gdf = process_geo_data(df_polygon)
df_top_crime, top_crime_list = process_crimes(df_clean)
gdf_plot_yearly = process_gdf_yearly(df_clean, gdf)
df_time_crime_percentage, df_time_crime_summary = crime_percent_time(df_clean,top_crime_list)
cases_by_month_year = crime_group(df_clean)
cases_by_date, cases_by_week, cases_by_month, cases_by_year = crime_cases_group(df_clean)



# --- Crime Seasonality Trend ---
fig_time_season = px.line(cases_by_month_year, x="Month", y="Cases", color="Year", markers=True)
fig_time_season.update_layout(
    title="Monthly Cases by Year (Seasonality)",
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    ),
    legend_title="Year",
    hovermode="x unified"
)

fig_crime_cases = go.Figure()
fig_crime_cases.add_trace(go.Scatter(x=cases_by_date["Date"], y=cases_by_date["Cases"],
                         mode="lines", name="By Date", visible=True))
fig_crime_cases.add_trace(go.Scatter(x=cases_by_week["Week"], y=cases_by_week["Cases"],
                         mode="lines", name="By Week", visible=False))
fig_crime_cases.add_trace(go.Scatter(x=cases_by_month["Month"], y=cases_by_month["Cases"],
                         mode="lines", name="By Month", visible=False))
fig_crime_cases.add_trace(go.Bar(x=cases_by_year["Year"], y=cases_by_year["Cases"],
                        name="By Year", visible=False))
# Add dropdown menu to toggle visibility
fig_crime_cases.update_layout(
    updatemenus=[
        dict(
            type="dropdown",
            x=0.1, y=1.15,
            buttons=[
                dict(label="Date", method="update",
                     args=[{"visible": [True, False, False, False]},
                           {"title": "Cases by Date"}]),
                dict(label="Week", method="update",
                     args=[{"visible": [False, True, False, False]},
                           {"title": "Cases by Week"}]),
                dict(label="Month", method="update",
                     args=[{"visible": [False, False, True, False]},
                           {"title": "Cases by Month"}]),
                dict(label="Year", method="update",
                     args=[{"visible": [False, False, False, True]},
                           {"title": "Cases by Year"}]),
            ]
        )
    ]
)


# --- Crime Density Map ---
fig_density = px.density_mapbox(
    df_clean,
    lat="Latitude",
    lon="Longitude",
    radius=10,
    hover_data=["Year", "Date"],
    color_continuous_scale="Viridis",
    mapbox_style="open-street-map",
    zoom=9,   # higher zoom = closer view
    center={"lat": 41.8781, "lon": -87.6298},  # Chicago coordinates
    height=600,
    title="Incident Density Map - Chicago Focus",
    animation_frame="Year"   # optional: play by year
)

# ---Choropleth Map Figure Preparation---
geojson = json.loads(gdf.copy().to_json())
fig_choropleth_overall = px.choropleth_mapbox(
    gdf_plot_yearly,
    geojson=geojson,
    locations="AREA_NUMBER",            
    featureidkey="properties.AREA_NUMBER",
    color="Crime/km2",
    animation_frame="Year",
    color_continuous_scale="Reds",
    mapbox_style="open-street-map",
    center={"lat": 41.828, "lon": -87.6298},
    zoom=9,
    range_color=(0, max(gdf_plot_yearly['Crime/km2'])),
    opacity=0.85,
    hover_name="COMMUNITY",
    hover_data={"Total Crime": True,
                "AREA_KM2": True},
    height=650,
    width=1000,
    title="Crime Density by Community Area Over The Years"
)

# ---Top Crime Horizontal Bar---
fig_top_crime_cum = px.bar(
    df_top_crime,
    x="Cummulative Count",
    y="Primary Type",
    animation_frame="Year",
    orientation="h",
    title="Cummulative Count for Chicago Crime Based on Type (Yearly, Top 11)",
)
fig_top_crime_cum.update_layout(
    xaxis_title="Crime Count",
    yaxis_title="Crime Type",
    height=800,
    width=1000
)
fig_top_crime_cum.update_xaxes(tickformat=",d")
fig_top_crime_cum.update_yaxes(categoryorder="total ascending")

# --- Crime Time of Day Percentage Occurence Bar ---
fig_crime_prc_bar = px.bar(
    df_time_crime_percentage,
    x="CrimePercentage",
    y="Primary Type",
    color="Time of Day",
    orientation="h",
    title="Crime Distribution by Time of Day",
    category_orders={
        "Time of Day": [
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
        ],
        "Primary Type":top_crime_list
    },
        color_discrete_map={
        "Morning": "#2ECC71",
        "Afternoon": "#F1C40f",
        "Evening": "#e67e22",
        "Night": "#213D97"
    }

)
fig_crime_prc_bar.update_layout(
    xaxis_title="Percentage of Crime (%)",
    yaxis_title="Crime Type",
    barmode="stack",
    height=600
)
fig_crime_prc_bar.for_each_trace(
    lambda t: t.update(
        name={
            "Morning": "Morning (5AM–9AM)",
            "Afternoon": "Afternoon (12PM–5PM)",
            "Evening": "Evening (5PM–9PM)",
            "Night": "Night (9PM–5AM)",
        }[t.name]
    )
)


# --- Crime Time of Day Summary PieChart ---
fig_crime_pie = px.pie(
    df_time_crime_summary,
    names="Time of Day",
    values="Crime Count",
    title="Crime Occurrences by Time of Day in Chicago 2015-2025",
    color="Time of Day",
        category_orders={
        "Time of Day": [
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
        ]
    },
    color_discrete_map={
        "Morning": "#2ECC71",
        "Afternoon": "#F1C40f",
        "Evening": "#e67e22",
        "Night": "#213D97"
    }
)
fig_crime_pie.update_traces(textinfo="label+percent")
fig_crime_pie.update_layout(height=600,width=600)


# ========== HOME PAGE ==========
if selected == "Project Data Overview":
    st.title("Chicago Crime Dataset - Exploratory Data Analysis")

    st.write("This streamlit application's main objective is to conduct an Exploratory Data Analysis (EDA) on crimes that are occuring "
    "in Chicago from from 2015 - 2015. The main data is obtained from the open source data provided by Chicago Data Portal.")
    st.write(f"The dataset contains {df_crime.shape[0]} rows and {df_crime.shape[1]} columns")
    # ===Dataset Overview===
    st.header("1. Dataset Overview")
    
    # Show column names and data types
    st.subheader("Column Names and Data Types")
    st.table(pd.DataFrame({"Column Name": df_crime.columns, "Data Type": df_crime.dtypes.astype(str).values}))

    # # ===Data Preview with Date Range Filter=== (Optional)
    # st.header("Data Preview with Date Range")

    # # Ensure date column is actually datetime object and get min and max
    # df_crime["Date"] = pd.to_datetime(df_crime["Date"])
    # min_date = df_crime["Date"].min().to_pydatetime()
    # max_date = df_crime["Date"].max().to_pydatetime()

    # date_range = st.slider(
    #     "Select date range",
    #     min_value=min_date,
    #     max_value=max_date,
    #     value=(min_date, max_date)
    # )

    # filtered_range_df = df_crime[
    #     (df_crime["Date"] >= date_range[0]) & (df_crime["Date"] <= date_range[1])
    # ]

    # st.write(f"Showing {len(filtered_range_df)} rows between {date_range[0]} and {date_range[1]}:")
    # st.dataframe(filtered_range_df.head(50))

    # ===Missing Values===
    st.header("Missing Values by Year")

    st.write("The following table displays the number of missing values for each column, grouped by year, to understand the trend of missing data " \
    "over different years. It is discovered that the amount of missing data is small compared to the whole dataset which will not distort the overall EDA when removed. " \
    "Also, this simplifies workflow - avoid complexity of imputing values which can introduce bias if not done carefully.")

    # Ensure date column is actually datetime object and get min and max
    df_crime["Date"] = pd.to_datetime(df_crime["Date"])

    if "Date" in df_crime.columns:
        missing_by_year = (
            df_crime.assign(Year=df_crime["Date"].dt.year)
            .groupby("Year")
            .apply(lambda x: x.isnull().sum())
        )
        #st.write(missing_by_year)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(missing_by_year.T, cmap="Reds", annot=True, fmt="d", ax=ax)
        ax.set_title("Missing Values by Year")
        st.pyplot(fig)

    # ===Duplicate Records Analysis===
    st.header("Duplicate Records")

    # Calculate metrics
    total_rows = len(df_crime)
    # keep=False ensures we count every instance of a duplicate for the count
    total_duplicate_rows = df_crime.duplicated(subset=['Case Number'], keep=False).sum()
    unique_duplicates = df_crime.duplicated(subset=['Case Number']).sum()
    duplicate_pct = (unique_duplicates / total_rows)

    # Display as metrics
    col1, col2 = st.columns(2)
    col1.metric("Duplicate Rows Found", unique_duplicates)
    col2.metric("Data Redundancy", f"{duplicate_pct:.4%}")

    if unique_duplicates > 0:
        st.warning(f"Found {unique_duplicates} redundant records. It is recommended to remove these before proceeding with the EDA.")
        
        # Show a sample of the duplicates
        if st.checkbox("Show sample of duplicate records"):
            sample_dups = df_crime[df_crime.duplicated(subset=['Case Number'], keep=False)].sort_values("Case Number")
            st.dataframe(sample_dups.head(20))

# ========== VISUALIZATION PAGE ==========

elif selected == "Visualization":

    # ===== Spatial Analysis =====
    st.title("🗺️ SPATIAL VISUALIZATION")
    # Crime Density (crimes/kg2) Choropleth Map of Chicago 
    st.header("Spatial Analysis: Choropleth")
    st.plotly_chart(fig_choropleth_overall, use_container_width=True)

    st.header("Spatial Analysis: Density Map")
    st.plotly_chart(fig_density, use_container_width=True)

    # ===== Time Series Analysis =====
    st.title("⏱️ TIME SERIES VISUALIZATION")
    
    st.header("Time Series Analysis: Cases Over Time")
    st.plotly_chart(fig_crime_cases, use_container_width=True)
    st.header("Time Series Analysis: Seasonality Trend")
    st.plotly_chart(fig_time_season, use_container_width=True)
    
    # ===== Graph & Plot Analysis =====
    st.title("📊 GRAPH & PLOT VISUALIZATION")
    st.header("Graph & Plot: Top 11 Crime Occurence Cummulative Occurence")
    st.plotly_chart(fig_top_crime_cum, use_container_width=True)
    st.header("Graph & Plot: Crime Occurence in Times of Day")
    st.plotly_chart(fig_crime_prc_bar, use_container_width=True)
    st.header("Graph & Plot: Crime Occurence Overall Percentage")
    st.plotly_chart(fig_crime_pie, use_container_width=True)


# ========== SUMMARY PAGE ==========
else:
    st.title(f"Summary page is still empty 🙂, be patient")

