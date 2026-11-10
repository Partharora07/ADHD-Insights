import pandas as pd

activity = pd.read_csv("data/raw/merged_activity_data.csv")
info = pd.read_csv("data/raw/patient_info.csv", sep=";")

activity["ID"] = (
    activity["source_file"]
    .str.extract(r"patient_activity_(\d+)")
    .astype(int)
)

matched = info[info["ID"].isin(activity["ID"].unique())]

print("Matched participants:", matched["ID"].nunique())

print("\nMatched label distribution:")
print(matched["ADHD"].value_counts().sort_index())

print("\nParticipants without activity data:")
print(
    info[~info["ID"].isin(activity["ID"].unique())]["ID"].tolist()
)

print("\nMatched participant demographics:")
print(matched.groupby("ADHD")["AGE"].describe())