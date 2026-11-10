
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/interim/labeled_activity_data.csv"

def main():
    print("Loading labelled activity dataset...")

    df = pd.read_csv(DATA_FILE)

    df["TIMESTAMP"] = pd.to_datetime(
        df["TIMESTAMP"],
        format="%m-%d-%Y %H:%M",
        errors="coerce"
    )

    print("\nDATASET OVERVIEW")
    print("-" * 40)
    print("Total rows:", len(df))
    print("Participants:", df["ID"].nunique())
    print("Missing timestamps:", df["TIMESTAMP"].isna().sum())

    # Participant-level recording statistics
    summary = df.groupby("ID").agg(
        total_records=("ACTIVITY", "count"),
        start_time=("TIMESTAMP", "min"),
        end_time=("TIMESTAMP", "max"),
        mean_activity=("ACTIVITY", "mean"),
        median_activity=("ACTIVITY", "median"),
        std_activity=("ACTIVITY", "std"),
        zero_readings=("ACTIVITY", lambda x: (x == 0).sum()),
        duplicate_timestamps=("TIMESTAMP", lambda x: x.duplicated().sum())
    ).reset_index()

    summary["recording_hours"] = (
        (summary["end_time"] - summary["start_time"])
        .dt.total_seconds() / 3600
    )

    summary["zero_percentage"] = (
        summary["zero_readings"] / summary["total_records"] * 100
    )

    print("\nRECORDING SUMMARY")
    print(summary.describe(include="all").to_string())

    print("\nSHORTEST RECORDINGS")
    print(summary.nsmallest(5, "recording_hours")[
        ["ID", "total_records", "recording_hours"]
    ].to_string(index=False))

    print("\nLONGEST RECORDINGS")
    print(summary.nlargest(5, "recording_hours")[
        ["ID", "total_records", "recording_hours"]
    ].to_string(index=False))

    print("\nDUPLICATE TIMESTAMPS:", 
          summary["duplicate_timestamps"].sum())

    output = ROOT / "data/interim/recording_summary.csv"
    summary.to_csv(output, index=False)

    print("\nSummary saved to:", output)


if __name__ == "__main__":
    main()
