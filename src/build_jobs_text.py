# src/build_jobs_text.py
# Requirements and description text of the open test-period jobs (for the "Why this job?" explanations)
import pandas as pd

from filters import open_jobs

OUT = "data/processed/jobs_text_open.parquet"
MAXC = 3000

jobs = pd.read_parquet("artifacts/jobs_open_union.parquet",
                       columns=["JobID", "WindowID", "StartDate", "EndDate"])
jobs = open_jobs(jobs, "2012-06-18", "2012-06-27")[["JobID", "WindowID"]].drop_duplicates()
print("open jobs:", len(jobs))

parts = []
for ch in pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                      usecols=["JobID", "WindowID", "Title", "Requirements", "Description"],
                      dtype={"Title": str, "Requirements": str, "Description": str},
                      chunksize=100000):
    ch = ch.merge(jobs, on=["JobID", "WindowID"], how="inner")
    for c in ["Title", "Requirements", "Description"]:
        ch[c] = ch[c].fillna("").str.slice(0, MAXC)
    parts.append(ch)

out = pd.concat(parts, ignore_index=True).sort_values(["WindowID", "JobID"])
print("rows:", len(out))
out.to_parquet(OUT, index=False, row_group_size=50000)
print("saved", OUT)