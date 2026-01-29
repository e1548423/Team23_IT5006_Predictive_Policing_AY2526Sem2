### Exploratory Data Analysis (EDA) App
#### How to run it on your own machine

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

5. Run the app
   ```
   streamlit run streamlit_app.py
   ```
   This will start a local server. You’ll see output like: Local URL: http://localhost:8501 Network URL: http://192.168.x.x:8501. Open the Local URL    in your browser to view the app.

#### Note: This application implements a local caching strategy. Upon the initial run (may take up to 5 minutes), the raw .csv dataset is fetched from Google Drive and serialized into the .parquet format. Subsequent launches prioritize this local Parquet cache, significantly reducing I/O overhead and memory usage by bypassing the 100MB+ cloud download.
