"""
EXOPLANET HABITABILITY EXPLORER — Part 2: The Web App
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Exoplanet Habitability Explorer", page_icon="🪐")

st.title(" Exoplanet Habitability Explorer")
st.write(
    "Describe a star system, and a machine learning model trained on "
    "real NASA exoplanet data will guess whether a planet around it "
    "could be a **habitable-zone candidate** — Earth-like temperature "
    "and roughly Earth-sized."
)
try:
    model=joblib.load("habitability_model.joblib")
    scaler=joblib.load("habitability_scaler.joblib")

except FileNotFoundError:
    st.error(
        " Model files not found! Run `python train_model.py` first "
        "to create them, then restart this app."
        )
    st.stop()

st.subheader("Describe the star and orbit")

col1, col2 = st.columns(2)
with col1:
    st_teff = st.slider("Star temperature (Kelvin)", 2500, 10000, 5778, step=50)
    st_rad = st.slider("Star radius (compared to our Sun = 1.0)", 0.1, 5.0, 1.0, step=0.1)
with col2:
    st_mass = st.slider("Star mass (compared to our Sun = 1.0)", 0.1, 3.0, 1.0, step=0.1)
    pl_orbper = st.slider("Orbital period (Earth days)", 1, 1000, 365, step=1)

st.caption("Defaults are set to our own Sun and Earth's orbit — try changing just one thing at a time!")

if st.button(" Predict habitability", type="primary"):
    input_df = pd.DataFrame([{
        "st_teff": st_teff,
        "st_rad": st_rad,
        "st_mass": st_mass,
        "pl_orbper": pl_orbper,
    }])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f" Habitable-zone candidate! (model confidence: {probability:.0%})")
        st.balloons()
    else:
        st.info(f" Probably not habitable-zone-friendly (model confidence it IS: {probability:.0%})")

    st.caption(
        "Remember: this is a simplified student project, not an official "
        "NASA habitability score. Real habitability depends on atmosphere, "
        "chemistry, and much more."
    )

st.divider()
st.caption(
    "Data source: NASA Exoplanet Archive (Planetary Systems Composite "
    "Parameters table)"
)
