
import pandas as pd
import numpy as np
from pathlib import Path

JOBS_PATH = "data/processed/jobs_clean.tsv"
OUTPUT_PATH = "data/processed/esco_review_sample.csv"

# 50 jobs distributed across 7 windows
targets = {
    str(i): (8 if i == 1 else 7)
    for i in range(1, 8)
}

rng = np.random.default_rng(42)
selected = {}

# Read in chunks to avoid loading 5GB into RAM
for chunk in pd.read_csv(
    JOBS_PATH,
    sep="\t",
    usecols=[
        "JobID", "WindowID", "Title",
        "Description", "Requirements"
    ],
    dtype=str,
    keep_default_na=False,
    chunksize=5000
):
    chunk["_random"] = rng.random(len(chunk))

    for window, group in chunk.groupby("WindowID"):
        if window not in targets:
            continue

        n = targets[window]
        candidates = group.nsmallest(n, "_random")

        if window in selected:
            candidates = pd.concat(
                [selected[window], candidates],
                ignore_index=True
            ).nsmallest(n, "_random")

        selected[window] = candidates.copy()

sample = pd.concat(
    selected.values(),
    ignore_index=True
)

sample = sample.drop(columns=["_random"])
sample = sample.sort_values(
    ["WindowID", "JobID"]
)

Path(OUTPUT_PATH).parent.mkdir(
    parents=True, exist_ok=True
)

sample.to_csv(OUTPUT_PATH, index=False)

print("Total sampled jobs:", len(sample))
print("\nJobs per window:")
print(sample["WindowID"].value_counts().sort_index())
print("\nSaved to:", OUTPUT_PATH)
