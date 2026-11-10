
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/interim/labeled_activity_data.csv"
OUTPUT_FILE = ROOT / "data/processed/participant_features.csv"


def longest_zero_bout(values):
    longest = 0
    current = 0

    for value in values:
        if value == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def extract_participant_features(group):
    group = group.sort_values("TIMESTAMP")

    activity = group["ACTIVITY"]
    timestamps = group["TIMESTAMP"]

    duration_hours = (
        (timestamps.max() - timestamps.min()).total_seconds() / 3600
        + 1 / 60
    )

    total_activity = activity.sum()
    mean_activity = activity.mean()

    active_minutes = (activity > 0).sum()
    zero_minutes = (activity == 0).sum()

    return pd.Series({
        "ADHD": group["ADHD"].iloc[0],

        "recording_duration_hours": duration_hours,
        "total_activity": total_activity,
        "activity_per_hour": total_activity / duration_hours,

        "mean_activity": mean_activity,
        "median_activity": activity.median(),
        "std_activity": activity.std(),
        "min_activity": activity.min(),
        "max_activity": activity.max(),

        "activity_90th_percentile": activity.quantile(0.90),
        "activity_95th_percentile": activity.quantile(0.95),

        "zero_activity_percentage": (
            zero_minutes / len(activity) * 100
        ),

        "active_minutes": active_minutes,
        "nonzero_activity_mean": (
            activity[activity > 0].mean()
            if active_minutes > 0 else 0
        ),

        "longest_zero_bout_minutes": longest_zero_bout(
            activity.tolist()
        )
    })


def main():
    print("Loading labelled activity data...")

    df = pd.read_csv(INPUT_FILE)

    df["TIMESTAMP"] = pd.to_datetime(
        df["TIMESTAMP"],
        format="%m-%d-%Y %H:%M",
        errors="raise"
    )

    df["ACTIVITY"] = pd.to_numeric(
        df["ACTIVITY"],
        errors="raise"
    )

    print("Extracting participant-level features...")

    features = (
        df.groupby("ID")
        .apply(extract_participant_features)
        .reset_index()
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_FILE, index=False)

    print("\nFEATURE EXTRACTION COMPLETE")
    print("-" * 40)
    print("Participants:", len(features))
    print("Total features:", len(features.columns) - 2)
    print("Output shape:", features.shape)

    print("\nFeature dataset preview:")
    print(features.head().to_string(index=False))

    print("\nLabel distribution:")
    print(features["ADHD"].value_counts().sort_index())

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
