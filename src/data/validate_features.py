
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data/processed/participant_features.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    print("\n1. DATASET SHAPE")
    print(df.shape)

    print("\n2. MISSING VALUES")
    print(df.isnull().sum())

    print("\n3. INFINITE VALUES")
    numeric = df.select_dtypes(include=np.number)
    print(np.isinf(numeric).sum())

    print("\n4. DUPLICATE PARTICIPANT IDs")
    print(df["ID"].duplicated().sum())

    print("\n5. LABEL DISTRIBUTION")
    print(df["ADHD"].value_counts())

    print("\n6. FEATURE STATISTICS")
    print(df.drop(columns=["ID", "ADHD"]).describe().T)

    print("\n7. CONSTANT FEATURES")
    features = df.drop(columns=["ID", "ADHD"])
    print(features.columns[features.nunique() <= 1].tolist())

    print("\n8. RECORDING DURATION RANGE")
    print(df["recording_duration_hours"].describe())

    print("\n9. CORRELATION WITH RECORDING DURATION")
    duration_corr = (
        df.drop(columns=["ID", "ADHD"])
        .corr(numeric_only=True)["recording_duration_hours"]
        .sort_values(ascending=False)
    )
    print(duration_corr)

    print("\nVALIDATION COMPLETED")


if __name__ == "__main__":
    main()
