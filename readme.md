# Exoplanet Habitable Zone Explorer

A machine learning web app that predicts whether an exoplanet might be in a star's "habitable zone," where it is the right temperature and size for life as we know it (Earth-like), using data from NASA's Exoplanet Archive.

## What it does

The app takes in four parameters about a star system:

- Star temperature (K)
- Star radius (relative to the Sun)
- Star mass (relative to the Sun)
- Planet's orbital period (days)

And returns a binary prediction and probability of being in a habitable zone based on the output of a Random Forest model trained on real exoplanets.

## Label definition

NASA does not provide a "habitability" label for exoplanets (and there is no consensus scientific measure). This project provides a simplified proxy label for demonstration purpose. A rough approximation of Earth-like temperatures and radius range, but not a real "habitability" index (as this relies on many other factors, like atmospheric conditions and stellar activity).

Critically, `pl_eqt` (temperature) and `pl_rade` (radius) — the two pieces of data needed to construct this label — are **not included in the features for the model**. Training the model on features used to construct the label would allow the model to "reverse-engineer" the label rather than learn a genuine relationship between a star's properties and the chance of containing a comfortable-size, comfortable-temperature planet.

## Data

Data source: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/), Planetary Systems Composite Parameters (`pscomppars`) table, queried directly via the public TAP API (no API key needed):

```
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_rade,pl_bmasse,pl_orbper,pl_eqt,st_teff,st_rad,st_mass+from+pscomppars&format=csv
```

Rows without a temperature, radius or any of the four training features are dropped. Habitability candidates are a tiny fraction of the dataset, so a stratified split is used. Both models are trained using `class_weight="balanced"` to address the issue of learning to predict the majority class ("not habitable").

## Model

Two classification models are trained and compared on a held-out 20% test split.The winner is chosen based on the **F1 score** of the "habitable candidate" class — with an imbalanced dataset, predicting the majority class all the time can lead to artificially high accuracy, so that metric isn't enough. In one run, the Random Forest got an F1 score of 0.62 on the habitable candidates class and won out as the saved model.

The winning model and its `StandardScaler` are saved to files with `joblib` and then loaded into the web app at runtime.

## Tech stack

- **Data**: `pandas`, NASA Exoplanet Archive TAP API via `requests`
- **Modeling**: `scikit-learn` (LogisticRegression, RandomForestClassifier, StandardScaler)
- **Persistence**: `joblib`

## Known limitations

- Habitability label is a simplified heuristic, not a scientific measure — it is supposed to demonstrate a realistic ML workflow (label engineering, leakage prevention, imbalanced classes handling) applied to real astronomical data, not make a scientific statement about exoplanets' properties.
- F1 around 0.6 represents a hard problem (predicting planet-scale features based on star-scale features) rather than a fine-tuned production-ready model.
- NASA Exoplanet Archive gets constantly updated as new exoplanets are discovered, therefore results will change every time `train_model.py` is run.

## Data source

NASA Exoplanet Archive, operated by Caltech/IPAC under contract with NASA: https://exoplanetarchive.ipac.caltech.edu/
