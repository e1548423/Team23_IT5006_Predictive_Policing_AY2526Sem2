# IT5006Group Chicago Crime Analysis

This repository contains IT5006 group project involving Data Analysis for Chicago crime activity from 2015 to 2025, developed by Group 23. Three main folders are included for this project:

1. EDA – Folder containing Exploratory Data Analysis conducted on Chicago Crime data.
2. ML – Folder containing model development and streamlit application.
3. GeneralizationTest - Folder containing generalization test result for the final model.

#### Note: Each folder has their own README.md file for further details.

# IT5006Group Chicago Crime Analysis
The goal of each project's milestone includes:
1. EDA          : Gain meaningful insights and uncover patterns to understand the nature of crime data in Chicago.
2. ML           : Develop a machine learning model that can be an additional tool to help law enforcement in crime policing.
3. Deployment   : Deploy the machine learning model into a web application as a Proof of Concept (PoC).

## File Structure

Note: The following file structure displays general but important files and folders. A much detailed file/folder structure can be viewed by the user.
```
├── EDA/
│ ├── 0. DatasetDownload.ipynb (Download dataset locally)
│ ├── 1. Exploratory Data Analysis.ipynb (Conduct full EDA)
│ ├── README.md
│ ├── streamlit-app.py (Streamlit application)
│ └── ProjectData/
├── GeneralizationTest/
│ ├── BostonPoliceDistricts/
│ ├── FeaturesDataset/
│ ├── LAPoliceDistricts/
│ ├── RawDataset/
│ ├── GeneralizationTest.ipynb (Conduct generalization test)
│ ├── ProcessBostonLA.ipynb (Preprocess dataset)
│ └── README.md
├── ML/
│ ├── App/ (Streamlit application for model deployment)
│ ├── Deploy_Render/ (Render configs for hosting FastAPI API)
│ ├── Model/ (Model development)
│ └── readme.md
├── ProjectData/ (Folder contains raw data from "0. DatasetDownload.ipynb")
├── .gitignore
└── README.md
```

## Streamlit Application
The deployed EDA and Machine Learning model can be viewed here:
- 🔍 **Exploratory Data Analysis**  [EDA Streamlit](https://appdeploytest-gepl8crjupkdwdcbadmtre.streamlit.app/)
- 🤖 **Machine Learning (Prediction Dashboard)**  [Crime Policing Steramlit](https://team23it5006predictivepolicingay2526sem2-sxfonxxmudo9cyzjct2au.streamlit.app/)
