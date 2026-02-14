# IT5006Group Chicago Crime Analysis

This repository contains IT5006 group project involving Exploratory Data Analysis for Chicago crime activity from 2015 to 2025 developed by Group 23.

## Abstract
The objective of the project is to gain insights and patterns for Chicago's crime activity from an intial assumption and curiosty which is later confirmed by creating visualizations based on geospatial, temporal, and other data. Through EDA, the team gained meaningful insights that strengthens the team's understanding of how crime operates in Chicago.

## Dataset
The datasets were acquired from Chicago's Data Portal include:

* Chicago crime dataset
https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data

* Chicago community area map boundary
https://data.cityofchicago.org/Facilities-Geographic-Boundaries/chicago-Community-areas/m39i-3ntz

## Objectives
* Identify spatial crime patterns across Chicago community areas
* Analyze temporal crime trends
* Examine the most frequent crime types over time
* Explore relationships between crime occurrence and arrest rates

## Key Findings
* High crime-density areas remain consistent over time
* Crime shows strong seasonality, peaking in mid-year, especially in July and August
* Theft, Battery, and Criminal Damage crimes dominate across all years compared to others
* Certain crime types are centralized in certain Community Area
* Arrest rates for the most occuring crime types remain in the lower end

## Code Structure
```
├── EDA/
│ ├── 0. DatasetDownload.ipynb
│ ├── 1. Exploratory Data Analysis.ipynb
│ ├── 1.1 EDA Summarized.ipynb
│ ├── README.md
│ ├── requirements.txt
│ ├── streamlit-app.py
│ ├── jsonvis/
│ └── ProjectData/
├── .gitignore
└── README.md
```
- ProjectData folder is created when running "0. DatasetDownload.ipynb" to store dataset from kaggle
- jsonvis folder is created when running "1. Exploratory Data Analysis.ipynb" to save figures locally

## How to Run

1. Clone or navigate to project directory
   ```
   cd /folder
   ```

2. Create a virtual environment (optional)
   ```
   python -m venv venv
   ```

3. Activate the virtual environment (Windows)
   ```
   venv\Scripts\activate
   ```

4. Install requirements
   ```
   pip install -r requirements.txt
   ```

5. run "0. DatasetDownload.ipynb" to download dataset files locally

6. run "1. Exploratory Data Analysis.ipynb" to see all of the visualizations made for this project

7. run streamlit app
   ```
   change directory to EDA by typing in the terminal "cd EDA"
   run the streamlit application by typing in the terminal: "streamlit run streamlit-app.py"
   ```
   This will start a local server. You’ll see output like: Local URL: http://localhost:8501 Network URL: http://192.168.x.x:8501. Open the Local URL    in your browser to view the app.


Note:
- The streamlit application implements a local caching strategy. Upon the initial run (may take up to 5 minutes), the raw .csv dataset is fetched from Google Drive and serialized into the .parquet format. Subsequent launches prioritize this local Parquet cache, significantly reducing I/O overhead and memory usage by bypassing the 100MB+ cloud download.
- The result of the EDA can be viewed from the streamlit live link: https://appdeploytest-gepl8crjupkdwdcbadmtre.streamlit.app/.
