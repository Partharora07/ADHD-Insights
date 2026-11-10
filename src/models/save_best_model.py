
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Project paths
ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "participant_features.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "selected_svm_model.pkl"


# Load participant-level data
df = pd.read_csv(INPUT_FILE)

# Same features used during model evaluation
FEATURES = [
    "mean_activity",
    "median_activity",
    "std_activity",
    "max_activity",
    "zero_activity_percentage",
    "longest_zero_bout_minutes"
]

X = df[FEATURES]
y = df["ADHD"]


# Build selected SVM pipeline
svm_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])


# Fit on all available participants for prototype use
svm_pipeline.fit(X, y)


# Save model and feature information together
artifact = {
    "model": svm_pipeline,
    "features": FEATURES,
    "target": "ADHD",
    "model_name": "SVM with RBF Kernel"
}

joblib.dump(artifact, MODEL_FILE)

print("\nSELECTED MODEL SAVED SUCCESSFULLY")
print("Model: SVM with RBF Kernel")
print("Participants used:", len(df))
print("Features:", len(FEATURES))
print("Saved at:", MODEL_FILE)
