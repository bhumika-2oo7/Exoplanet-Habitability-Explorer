"""
EXOPLANET HABITABILITY EXPLORER — Part 1: Train the Model
--------------------------------------------------------------
The app takes in four parameters about a star system:

Star temperature (K)
Star radius (relative to the Sun)
Star mass (relative to the Sun)
Planet's orbital period (days)
And returns a binary prediction and probability of being in a habitable zone based on the output of a Random Forest model trained on real exoplanets.
A bigger AI project.
"""

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
query = (
    "select pl_name,pl_rade,pl_bmasse,pl_orbper,pl_eqt,"
    "st_teff,st_rad,st_mass from pscomppars"
)
response = requests.get(TAP_URL, params={"query": query, "format": "csv"}, timeout=60)
response.raise_for_status()

with open("exoplanets_raw.csv", "wb") as f:
    f.write(response.content)

df = pd.read_csv("exoplanets_raw.csv")
print(f"Downloaded {len(df)} exoplanets!\n")

df = df.dropna(subset=["pl_eqt", "pl_rade"])
df["is_habitable_candidate"] = (
    (df["pl_eqt"].between(200, 320)) & (df["pl_rade"].between(0.5, 1.6))
).astype(int)

print("How many planets fall into each group?")
print(df["is_habitable_candidate"].value_counts(), "\n")


feature_columns = ["st_teff", "st_rad", "st_mass", "pl_orbper"]
df_clean = df.dropna(subset=feature_columns)

X = df_clean[feature_columns]
y = df_clean["is_habitable_candidate"]

print(f"Training on {len(df_clean)} planets with complete data.")
print("Label balance in this final dataset:")
print(y.value_counts(), "\n")



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
}

results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)
    results[model_name] = {"model": model, "accuracy": accuracy, "f1": f1}
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy:.1%}   F1 score: {f1:.2f}")
    print(classification_report(y_test, predictions, zero_division=0))

best_name = max(results, key=lambda name: results[name]["f1"])
best_model = results[best_name]["model"]
print(f" Winner: {best_name} (F1 = {results[best_name]['f1']:.2f})")

joblib.dump(best_model, "habitability_model.joblib")
joblib.dump(scaler, "habitability_scaler.joblib")
print("\n Saved the trained model and scaler to disk.")
print("   Now run the web app with")
