
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/participant_features.csv"
OUTPUT_DIR = ROOT / "reports/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

sns.set_theme(style="whitegrid")

# 1. Label distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="ADHD")
plt.title("Participant Label Distribution")
plt.xlabel("ADHD Dataset Label")
plt.ylabel("Number of Participants")
plt.savefig(OUTPUT_DIR / "label_distribution.png", dpi=300)
plt.close()

# Candidate features
features = [
    "mean_activity",
    "median_activity",
    "std_activity",
    "max_activity",
    "activity_90th_percentile",
    "activity_95th_percentile",
    "zero_activity_percentage",
    "nonzero_activity_mean",
    "longest_zero_bout_minutes"
]

# 2-10. Individual feature distributions
for feature in features:
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="ADHD",
        y=feature
    )

    plt.title(f"{feature} by Dataset Label")
    plt.xlabel("ADHD Dataset Label")
    plt.ylabel(feature)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{feature}_comparison.png",
        dpi=300
    )
    plt.close()

# 11. Correlation heatmap
plt.figure(figsize=(12, 9))

correlation = df[features].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Activity Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_correlation.png",
    dpi=300
)

plt.close()

print("EDA COMPLETED SUCCESSFULLY!")
print("Graphs saved in:", OUTPUT_DIR)
