### Exploratory Data Analysis (EDA) App
#### How to run streamlit on your own machine

1. Activate the virtual environment (Windows)
   ```
   venv\Scripts\activate
   ```

2. Navigate to EDA folder by typing "cd EDA" in the terminal

3. Run the app
   ```
   streamlit run streamlit-app.py
   ```
   This will start a local server. You’ll see output like: Local URL: http://localhost:8501 Network URL: http://192.168.x.x:8501. Open the Local URL    in your browser to view the app.

#### Note: This application implements a local caching strategy. Upon the initial run (may take up to 5 minutes), the raw .csv dataset is fetched from Google Drive and serialized into the .parquet format. Subsequent launches prioritize this local Parquet cache, significantly reducing I/O overhead and memory usage by bypassing the 100MB+ cloud download.
