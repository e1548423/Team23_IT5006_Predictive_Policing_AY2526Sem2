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

## Repository Structure
.
├── EDA/
│   ├── 0. DatasetDownload.ipynb
│   ├── 1. Exploratory Data Analysis.ipynb
│   ├── 1.1 EDA Summarized.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── streamlit-app.py
│   ├── jsonvis/
│   └── ProjectData/
├── .gitignore
└── README.md

* ProjectData folder is created when running "0. DatasetDownload.ipynb" to store dataset from kaggle
* jsonvis folder is created when running "1. Exploratory Data Analysis.ipynb" or "1.1 EDA Summarized.ipynb" to save figures locally


## How to Run

Users are able to view the complete process and visualization the team produced by following these steps:
1. Creating a virtual environment complete with the libraries included in requirements.txt, details can be read from the README.md file inside EDA folder
2. run 0. DatasetDownload.ipynb
3. run 1. Exploratory Data Analysis.ipynb

The result of the EDA has been summarized and can be viewed from the streamlit live link: https://appdeploytest-gepl8crjupkdwdcbadmtre.streamlit.app/.