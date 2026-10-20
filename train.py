"""
Train the heart disease model.

Run:  python train.py

What it does
1. Loads data/heart.csv (downloads it from the UCI repository the first time).
2. Cleans the data (0 cholesterol / 0 blood pressure are missing values, not real readings).
3. Compares several algorithms with 5-fold cross-validation on the training set.
4. Evaluates the best one on a held-out test set.
5. Saves ONE pipeline (preprocessing + model) to model/heart_model.joblib
   and the evaluation results to model/metrics.json.

Fixes compared with the original notebook:
- Oldpeak is kept as a decimal (the notebook did astype(int), turning 1.5 into 1).
- Zero cholesterol / blood pressure are imputed instead of being treated as real values.
- Preprocessing lives inside the saved pipeline, so the app cannot encode inputs differently
  from training (no hand-built dummy columns).
- The model outputs a probability, not just a 0/1 label.
"""
import io
import json
import ssl
import urllib.request
from pathlib import Path

import certifi

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "heart_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET = "HeartDisease"
RANDOM_STATE = 42

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.{}.data"
UCI_SITES = ["cleveland", "hungarian", "switzerland", "va"]
UCI_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "num"]


def download_dataset():
    """Build heart.csv (same schema as the Kaggle 'Heart Failure Prediction' dataset) from UCI."""
    print("Downloading the UCI heart disease data (4 hospitals)...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    frames = []
    for site in UCI_SITES:
        with urllib.request.urlopen(UCI_BASE.format(site), context=ctx, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        frames.append(pd.read_csv(io.StringIO(text), header=None, names=UCI_COLS, na_values="?"))
    raw = pd.concat(frames, ignore_index=True)

    df = pd.DataFrame({
        "Age": raw["age"],
        "Sex": raw["sex"].map({1: "M", 0: "F"}),
        "ChestPainType": raw["cp"].map({1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}),
        "RestingBP": raw["trestbps"],
        "Cholesterol": raw["chol"],
        "FastingBS": raw["fbs"],
        "RestingECG": raw["restecg"].map({0: "Normal", 1: "ST", 2: "LVH"}),
        "MaxHR": raw["thalach"],
        "ExerciseAngina": raw["exang"].map({1: "Y", 0: "N"}),
        "Oldpeak": raw["oldpeak"],
        "ST_Slope": raw["slope"].map({1: "Up", 2: "Flat", 3: "Down"}),
        TARGET: (raw["num"] > 0).astype(int),
    })
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {DATA_PATH}")


def load_data():
    if not DATA_PATH.exists():
        download_dataset()
    df = pd.read_csv(DATA_PATH)
    # A cholesterol or blood pressure of 0 is physically impossible -> it means "not measured".
    for col in ["Cholesterol", "RestingBP"]:
        df[col] = df[col].replace(0, np.nan)
    df["FastingBS"] = df["FastingBS"].astype("Int64").astype(str).replace("<NA>", np.nan)
    df = df.dropna(subset=[TARGET]).drop_duplicates()
    return df


def build_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


CANDIDATES = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                            random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "SVM (RBF)": CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False),
}


def main():
    df = load_data()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    print(f"Dataset: {len(df)} rows | heart disease rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    print("\n5-fold cross-validation (ROC AUC) on the training set:")
    for name, clf in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = float(scores.mean())
        print(f"  {name:<20} {scores.mean():.3f} ± {scores.std():.3f}")

    best_name = max(cv_scores, key=cv_scores.get)
    best = Pipeline([("prep", build_preprocessor()), ("model", CANDIDATES[best_name])])
    best.fit(X_train, y_train)

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": best_name,
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(df)),
        "n_test": int(len(y_test)),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "cv_roc_auc": cv_scores,
    }
    print(f"\nBest model: {best_name}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  test {k:<9} {metrics[k]:.3f}")

    # Refit on all data for the final saved model.
    best.fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
