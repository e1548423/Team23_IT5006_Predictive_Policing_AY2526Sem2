# IT5006Group Chicago Crime Analysis

This repository contains IT5006 group project involving Data Analysis for Chicago crime activity from 2015 to 2025, developed by Group 23. The work is organized based on the project's milestones, categorized into two main Folders:

1. Exploratory Data Analysis (EDA) – located in the EDA/ folder.
2. Machine Learning Model (ML) – located in the ML/ folder.

Each folder has their own README.md file for further details.

# IT5006Group Chicago Crime Analysis
The goal of each milestone includes:
1. EDA   : Gain meaningful insights and uncover patterns to understand the nature of crime data in Chicago.
2. ML    : Develop a machine learning model that can be an additional tool to help law enforcement in crime policing.

## Code Structure
```
├── EDA/ *Folder contains essential files to conduct EDA with a provided streamlit app deployment.
│ ├── 0. DatasetDownload.ipynb
│ ├── 1. Exploratory Data Analysis.ipynb
│ ├── 1.1 EDA Summarized.ipynb
│ ├── README.md
│ ├── requirements.txt
│ ├── streamlit-app.py
│ ├── jsonvis/
│ └── ProjectData/
├── ML/ *Folder contains notebooks for ML training and analysis.
│ ├── Model
│ │ ├── deployment *pipeline and metada of the final chosen ML model
│ │ ├── Inference_Engine_UI.ipynb
│ │ └── Crime_Prediction_Training (Violent Crime).ipynb
│ ├── README.md
│ └── requirements.txt
├── ProjectData/ *Folder contains raw data after running 0. DatasetDownload.ipynb
├── .gitignore
└── README.md
```