# Generalization Test Introduction
With the final model now acquired, a generalization test needs to be carried on to understand the model's capability to predict crime occurence in other areas. Since the model was developed 
using Chicago's crime data, crime data outside of chicago will be tested.

The generalization test initially was to be conducted using data provided by NIBRS. But it was later discovered that NIBRS data lacked the Latitude and Longitude for where the crime occured. This is important since the information regarding the exact detail of where the crime occured will provide a much finer area monitoring through h3 address.

Thus, the model will be tested using crime data from Los Angeles and Boston. Both crime data provides the exact time, date, and location of where the crime occured.

###### *H3 is a global grid indexing system. Grid systems use a shape, like rectangles or triangles, to tessellate a surface, which in this case is the Earth's surface. Source: [Data Bricks H3 Geospatial](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-h3-geospatial-functions#:~:text=H3%20is%20a%20global%20grid,about%20the%20origins%20of%20H3.)

# Crime Data Sources

- **Los Angeles Crime Data (2020–2024)**: [Crime Data from 2020 to 2024](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-2024/2nrs-mtv8/about_data)
- **Boston Crime Incident Reports (2015–Present)**: [Crime Incident Reports – New System](https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system/resource/b973d8cb-eeb2-4e7e-99da-c92938efc9c0)

Both crime dataset needed to be preprocessed first to ensure consistent features with the ML model. For the generalization test, the crime data spans from 2022 - 2025. The finalized data covering Boston spans from 2022 - 2024 while LA data spans from 2023 - 2025. This ensures that the time period matches with the Chicago dataset that was used to train the ML model.


# Folder Structure

```
.
├── BostonPoliceDistricts
│ └── BostonPoliceDistricts.geojson
│
├── LAPoliceDistricts
│ └── LAPD_Map.geojson
│
├── RawDataset # Acquired raw datasets (Boston & LA)
│ ├── BostonCrime.parquet
│ └── LACrime.parquet
│
├── FeatureDataset # Preprocessed feature datasets
│ ├── BostonCrimeFeatures.parquet
│ └── LACrimeFeatures.parquet
│
├── GeneralizationTest.ipynb # Main notebook for model generalization testing
├── ProcessBostonLA.ipynb # Data preprocessing pipeline for Boston & LA
├── VisualizationBostonLA.ipynb # Exploratory analysis & visualization
│
└── xgb_calibrated_pipeline.joblib # Pre-trained XGBoost calibrated pipeline
```