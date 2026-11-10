from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

info = pd.read_csv(ROOT / "data/raw/patient_info.csv", sep=";")
activity = pd.read_csv(ROOT / "data/raw/merged_activity_data.csv")

activity["ID"] = (
    activity["source_file"]
    .str.extract(r"patient_activity_(\d+)")
    .astype(int)
)

matched_ids = activity["ID"].unique()
matched = info[info["ID"].isin(matched_ids)].copy()
excluded = info[~info["ID"].isin(matched_ids)].copy()

print("\nAGE CATEGORY DISTRIBUTION")
print(pd.crosstab(matched["AGE"], matched["ADHD"], margins=True))

print("\nSEX DISTRIBUTION")
print(pd.crosstab(matched["SEX"], matched["ADHD"], margins=True))

print("\nAGE BY ADHD CLASS")
print(matched.groupby("ADHD")["AGE"].value_counts().sort_index())

print("\nEXCLUDED PARTICIPANTS: LABEL DISTRIBUTION")
print(excluded["ADHD"].value_counts().sort_index())

print("\nEXCLUDED PARTICIPANTS: AGE DISTRIBUTION")
print(excluded["AGE"].value_counts().sort_index())

print("\nDEMOGRAPHIC ANALYSIS COMPLETE")