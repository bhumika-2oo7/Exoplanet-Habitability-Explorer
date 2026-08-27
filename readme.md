# 🪐 Exoplanet Habitability Explorer

A small machine learning project where I used real exoplanet data from NASA to explore whether a planet could be considered a possible habitability candidate.

## What is this project?

I wanted to try building an ML project using a real scientific dataset instead of a small sample dataset.

For this project, I used data from the **NASA Exoplanet Archive** and created my own simple rule for identifying possible habitable-zone candidates.

The project then:

* Downloads exoplanet data from NASA's Exoplanet Archive.
* Cleans the dataset and selects the features needed for the model.
* Creates a simple target label based on approximate Earth-like conditions.
* Trains two models: **Logistic Regression** and **Random Forest**.
* Compares their performance using the **F1 score**.
* Saves the better-performing model.
* Uses the trained model to make predictions for new inputs.

## Why I made it

I wanted to understand what a complete ML project looks like when working with real-world data.

I also wanted to learn about some things that I had not worked with much before, such as:

* Getting data from a public API.
* Cleaning real datasets.
* Creating a target variable when the dataset doesn't directly provide one.
* Dealing with an imbalanced dataset.
* Comparing different ML models.
* Connecting a trained model to a web interface.

This project is mainly a learning project, so the habitability rule I use is **simplified** and should not be treated as an actual scientific method for determining whether an exoplanet can support life.

## How the ML part works

I used information such as:

* Star temperature
* Star radius
* Star mass
* Planet orbital period

I used these features to train the models.

The target label is created separately using my simplified habitability conditions. I did not include the exact values used to create that label as model features, because that would make the prediction artificially easy.

I tested both Logistic Regression and Random Forest and compared their F1 scores before choosing the model used by the application.

## Running the project locally

### 1. Install the requirements

Make sure Python 3 is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Train the model

Run:

```bash
python train_model.py
```

This downloads the NASA dataset, prepares the data, trains the models and saves the selected model.

### 3. Start the application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## What I learned

This project helped me understand that building an ML project is more than just training a model.

I learned how to:

* Work with a real dataset instead of a prepared tutorial dataset.
* Get data from NASA's public API.
* Clean and select useful columns from a large dataset.
* Create labels from conditions I defined.
* Think about data leakage.
* Use F1 score when accuracy isn't enough.
* Compare different ML models.
* Save and load a trained model.
* Connect a machine learning model to a simple web interface.

## Data source

**NASA Exoplanet Archive**

I used the Planetary Systems Composite Parameters (PSCompPars) table through NASA's public TAP API.

https://exoplanetarchive.ipac.caltech.edu/

## Project structure

```text
Exoplanet-Habitability-Explorer/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── ...
```

## Note

This project is an educational ML experiment. The definition of "habitable" used here is intentionally simplified and does not represent the full scientific process used by astronomers to evaluate exoplanet habitability.
