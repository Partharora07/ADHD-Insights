
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ACTIVITY_FILE = ROOT / "data/raw/merged_activity_data.csv"
INFO_FILE = ROOT / "data/raw/patient_info.csv"
OUTPUT_FILE = ROOT / "data/interim/labeled_activity_data.csv"


def main():
    print("Loading datasets...")

    activity = pd.read_csv(ACTIVITY_FILE)
    info = pd.read_csv(INFO_FILE, sep=";")

    # Extract participant ID from source filename
    activity["ID"] = (
        activity["source_file"]
        .str.extract(r"patient_activity_(\d+)")
        .astype(int)
    )

    # Keep only the required participant information
    labels = info[["ID", "ADHD"]].copy()

    # Ensure IDs and labels have numeric types
    labels["ID"] = pd.to_numeric(labels["ID"], errors="raise")
    labels["ADHD"] = pd.to_numeric(labels["ADHD"], errors="raise")

    # Check participant IDs are unique
    if labels["ID"].duplicated().any():
        raise ValueError("Duplicate participant IDs in patient_info.csv")

    # Merge activity records with participant labels
    result = activity.merge(
        labels,
        on="ID",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    # Validate matching
    unmatched = result[result["_merge"] != "both"]

    if not unmatched.empty:
        raise ValueError(
            f"Found {unmatched['ID'].nunique()} unmatched participants."
        )

    result.drop(columns=["_merge"], inplace=True)

    # Save labelled data
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("\nLABEL ATTACHMENT COMPLETE")
    print("-" * 40)
    print("Total records:", len(result))
    print("Unique participants:", result["ID"].nunique())
    print("Missing ADHD labels:", result["ADHD"].isna().sum())

    print("\nParticipant label distribution:")
    print(
        result[["ID", "ADHD"]]
        .drop_duplicates()["ADHD"]
        .value_counts()
        .sort_index()
    )

    print("\nOutput saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
