import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import requests

# ─── CONFIG ────────────────────────────────────────────────────────────

# 1) Your Google Drive file ID for the big model:
DRIVE_FILE_ID = "1IO1dSxxuYlJyeTlQltoQKyKA55Pw8fE3"

# 2) Your GitHub raw URL for the small columns pickle:
COLUMNS_URL = (
    "https://raw.githubusercontent.com/YourUserName/YourRepoName/"
    "main/path/to/model_columns.pkl"
)

MODEL_PATH   = "pollution_model.pkl"
COLUMNS_PATH = "model_columns.pkl"


# ─── DRIVE DOWNLOADER ──────────────────────────────────────────────────

def download_drive_file(file_id: str, dest: str):
    """
    Download a Google Drive file by ID, handling the “large file” confirm token.
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    # Initial request
    resp = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
    if token:
        # Confirm and re-request
        resp = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(32_768):
            if chunk:
                f.write(chunk)


# ─── DOWNLOAD & LOAD ───────────────────────────────────────────────────

def load_model_and_columns():
    # 1) Download model if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏬ Downloading model from Google Drive…"):
            download_drive_file(DRIVE_FILE_ID, MODEL_PATH)

    # 2) Download columns if missing
    if not os.path.exists(COLUMNS_PATH):
        with st.spinner("⏬ Downloading model columns from GitHub…"):
            r = requests.get(COLUMNS_URL)
            r.raise_for_status()
            with open(COLUMNS_PATH, "wb") as f:
                f.write(r.content)

    # 3) Load both pickles
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH, "rb") as f:
        model_col = pickle.load(f)

    return model, model_col


# ─── INITIALIZE ────────────────────────────────────────────────────────

model, model_col = load_model_and_columns()

st.title('💧 Water Quality Prediction')
st.write('This model predicts the water quality based on the following parameters: Year and Station ID.')


col1, col2 = st.columns(2)
with col1:
    year_input = st.number_input('Year', min_value=2000, max_value=2050, value=2024, help="Select the year for prediction.")
with col2:
    station_id = st.text_input('Station ID', value='1', placeholder='e.g. 1', help="Enter the station ID 1-22.")

if st.button('🔮 Predict'):
    if not station_id:
        st.warning("Please enter the station ID")
    else:
        input_df= pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])
        for col in model_col:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_col]
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        st.subheader(f"📊 Predicted Pollutant Levels for Station {station_id} in {year_input}")

        predicted_df = pd.DataFrame({
            'Parameter': pollutants,
            'Predicted Value (mg/L)': [f"{val:.2f}" for val in predicted_pollutants]
        })
        st.table(predicted_df.set_index('Parameter'))

                
        st.markdown("---")
        st.subheader("Interpretation of Water Quality Parameters:")
        st.markdown("""
        * **$NH_4$ (Ammonium)**: High levels may indicate pollution from wastewater or agricultural runoff and can be toxic to aquatic life.  
        _(Typical Acceptable Limit: < 0.5 mg/L for drinking water)_

        * **BSK5 (BOD5)**: A high BSK5 indicates a lot of organic pollution, consuming oxygen and harming aquatic life.  
        _(Typical Acceptable Limit: < 3 mg/L for surface water)_

        * **Suspended Solids**: High levels reduce light penetration, can clog fish gills, or smother eggs.  
        _(Typical Acceptable Limit: < 25 mg/L for surface water)_

        * **$O_2$ (Dissolved Oxygen)**: Essential for aquatic life. Low levels ($<5~mg/L$) stress or kill aquatic organisms.  
        _(Typical Acceptable Limit: > 5 mg/L)_

        * **$NO_3$ (Nitrate)**: In excess, promotes algae growth (eutrophication) and can harm aquatic ecosystems and drinking water safety.  
        _(Typical Acceptable Limit: < 10 mg/L as NO₃-N for drinking water)_

        * **$NO_2$ (Nitrite)**: Toxic to aquatic organisms, even at low concentrations, indicating a breakdown in nitrogen processing.  
        _(Typical Acceptable Limit: < 0.1 mg/L for drinking water)_

        * **$SO_4$ (Sulfate)**: Generally not harmful in low concentrations, but can affect taste and promote corrosion.  
        _(Typical Acceptable Limit: < 250 mg/L for drinking water)_

        * **$PO_4$ (Phosphate)**: Excess leads to algal blooms and eutrophication, causing oxygen depletion and fish kills.  
        _(Typical Acceptable Limit: < 0.1 mg/L for surface water)_

        * **Cl (Chloride)**: High concentrations affect drinking water taste and harm freshwater organisms.  
        _(Typical Acceptable Limit: < 250 mg/L for drinking water)_
        """)


        st.info("The scoring is based on general acceptable limits and environmental impact. Specific regulations may vary.")
