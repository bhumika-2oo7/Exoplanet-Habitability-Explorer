# 🪐 Exoplanet Habitability Explorer

An AI/Machine Learning web app built.

## What it does

- Downloads real data on thousands of confirmed exoplanets from **NASA's
  Exoplanet Archive**.
- Defines a simplified "habitable-zone candidate" rule based on real
  astrophysics: Earth-like temperature + roughly Earth-sized.
- Trains and compares **two machine learning models** (Logistic
  Regression and Random Forest) to predict this from a star's
  temperature, radius, mass, and the planet's orbital period —
  *without* letting the model see the exact numbers used to build the
  label, to avoid "cheating."
- Picks the better-performing model and saves it.
- Serves it through an interactive **Streamlit web app** where anyone
  can describe a star system and get a live prediction.

## Why I made this

I wanted to go further: use a bigger real dataset, compare
multiple models properly, handle imbalanced data the right way, and
ship something people can actually click around in — not just read
terminal output.

## How to run it

1. Install Python 3.
2. In this folder, install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Train the model (this downloads NASA data and takes a minute):
   ```
   python train_model.py
   ```
4. Launch the web app:
   ```
   streamlit run app.py
   ```
   Your browser should open automatically to the app.


## What I learned

- How to pull large real datasets from a public science API
- How to engineer a label myself when no "answer key" exists
- Why using the same data to create a label AND train on it can "leak"
  the answer and produce fake-looking results
- How to compare multiple ML models fairly using F1 score, not just
  accuracy, especially with imbalanced data
- How to turn a Python script into an interactive web app with Streamlit

## Data source

[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) —
Planetary Systems Composite Parameters (PSCompPars) table, accessed via
their free public TAP API (no key required).
