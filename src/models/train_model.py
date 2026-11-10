
from pathlib import Path

import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC


# ==========================================
# 1. PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "participant_features.csv"

OUTPUT_DIR = ROOT / "reports" / "model_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("\nDATA LOADED")
print("Participants:", len(df))


# ==========================================
# 3. FEATURE SELECTION
# ==========================================

feature_cols = [
    "mean_activity",
    "median_activity",
    "std_activity",
    "max_activity",
    "zero_activity_percentage",
    "longest_zero_bout_minutes"
]

X = df[feature_cols].copy()
y = df["ADHD"]

if X.isnull().any().any():
    print("Missing values will be imputed inside each fold.")

if y.isnull().any():
    raise ValueError("Target contains missing values.")

if y.nunique() != 2 or y.value_counts().min() < 5:
    raise ValueError("Insufficient data for stratified 5-fold CV.")

print("\nClass distribution:")
print(y.value_counts())


# ==========================================
# 4. REPEATED CROSS VALIDATION
# ==========================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {

    "Dummy Baseline": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="most_frequent"))
    ]),

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ))
    ]),

    "SVM RBF": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=3,
            random_state=42
        ))
    ])
}


# ==========================================
# 6. EVALUATION METRICS
# ==========================================

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

results = []

for name, model in models.items():

    print(f"\nEvaluating: {name}")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise"
    )

    row = {"model": name}

    for metric in scoring:

        values = scores[f"test_{metric}"]

        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

    results.append(row)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="roc_auc_mean",
    ascending=False
)

results_df.to_csv(
    OUTPUT_DIR / "repeated_model_comparison.csv",
    index=False
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 70)
print("REPEATED MODEL EVALUATION COMPLETE")
print("=" * 70)

print("\nParticipants:", len(df))
print("Features:", len(feature_cols))
print("Validation: 5 folds x 5 repeats = 25 evaluations per model")

print("\nMODEL COMPARISON:\n")
print(results_df.round(3).to_string(index=False))

print("\nSaved to:")
print(OUTPUT_DIR / "repeated_model_comparison.csv")

print("\nEvaluation completed successfully!")
