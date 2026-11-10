
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/interim/labeled_activity_data.csv"
OUTPUT_FILE = ROOT / "data/interim/timestamp_gap_summary.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    df["TIMESTAMP"] = pd.to_datetime(
        df["TIMESTAMP"],
        format="%m-%d-%Y %H:%M",
        errors="coerce"
    )

    df = df.sort_values(["ID", "TIMESTAMP"])

    # Calculate time difference between consecutive readings
    df["time_gap_minutes"] = (
        df.groupby("ID")["TIMESTAMP"]
        .diff()
        .dt.total_seconds() / 60
    )

    # Exclude the first reading of each participant
    gaps = df.dropna(subset=["time_gap_minutes"]).copy()

    print("\nTIME GAP DISTRIBUTION")
    print("-" * 40)
    print(gaps["time_gap_minutes"].describe())

    print("\nINTERVAL FREQUENCIES")
    print(gaps["time_gap_minutes"].value_counts().head(15))

    # Summarise per participant
    summary = gaps.groupby("ID").agg(
        median_gap=("time_gap_minutes", "median"),
        mean_gap=("time_gap_minutes", "mean"),
        max_gap=("time_gap_minutes", "max"),
        intervals_over_1_min=(
            "time_gap_minutes",
            lambda x: (x > 1).sum()
        ),
        intervals_over_60_min=(
            "time_gap_minutes",
            lambda x: (x > 60).sum()
        )
    ).reset_index()

    print("\nPARTICIPANTS WITH MOST GAPS")
    print(
        summary.sort_values(
            "intervals_over_1_min",
            ascending=False
        ).head(10).to_string(index=False)
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved summary to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
