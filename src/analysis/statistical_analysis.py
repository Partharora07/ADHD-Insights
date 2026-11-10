
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "participant_features.csv"
OUTPUT_DIR = ROOT / "reports" / "statistical_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Exclude ID, target label and constant features
excluded = ["ID", "ADHD", "min_activity"]

features = [
    col for col in df.select_dtypes(include="number").columns
    if col not in excluded and df[col].nunique() > 1
]

group_0 = df[df["ADHD"] == 0]
group_1 = df[df["ADHD"] == 1]

results = []

for feature in features:
    x = group_1[feature].dropna()
    y = group_0[feature].dropna()

    if len(x) == 0 or len(y) == 0:
        continue

    test = mannwhitneyu(x, y, alternative="two-sided")

    # Cliff's delta: effect size (group 1 compared with group 0)
    delta = (2 * test.statistic / (len(x) * len(y))) - 1

    results.append({
        "feature": feature,
        "label_0_median": y.median(),
        "label_0_IQR": y.quantile(0.75) - y.quantile(0.25),
        "label_1_median": x.median(),
        "label_1_IQR": x.quantile(0.75) - x.quantile(0.25),
        "U_statistic": test.statistic,
        "p_value": test.pvalue,
        "cliffs_delta": delta
    })

result_df = pd.DataFrame(results)

# Benjamini-Hochberg multiple testing correction
m = len(result_df)

if m:
    sorted_indices = result_df["p_value"].sort_values().index
    sorted_p = result_df.loc[sorted_indices, "p_value"]

    adjusted = (sorted_p * m / range(1, m + 1)).clip(upper=1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1]

    result_df["p_adjusted_BH"] = adjusted.reindex(result_df.index)
    result_df["significant_0.05"] = result_df["p_adjusted_BH"] < 0.05

result_df = result_df.sort_values("p_adjusted_BH")

output_file = OUTPUT_DIR / "feature_statistics.csv"
result_df.to_csv(output_file, index=False)

print("\nSTATISTICAL ANALYSIS COMPLETE")
print("--------------------------------")
print("Participants:", len(df))
print("Label 0:", len(group_0))
print("Label 1:", len(group_1))
print("Features tested:", len(result_df))
print("Significant features (BH adjusted p < 0.05):",
      result_df["significant_0.05"].sum())

print("\nFEATURE RESULTS:")
print(result_df.to_string(index=False))

print("\nSaved to:", output_file)
