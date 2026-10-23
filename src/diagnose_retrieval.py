import numpy as np
import pandas as pd
from split_utils import load_apps, temporal_split

jobs = (pd.read_parquet("artifacts/jobs_keys.parquet")
          .drop_duplicates(["JobID", "WindowID"]))
users = (pd.read_parquet("artifacts/users_keys.parquet")
           .drop_duplicates(["UserID", "WindowID"]))
_, val, _ = temporal_split(load_apps())

v = (val.merge(users.rename(columns={"State": "user_state"}),
               on=["UserID", "WindowID"], how="left")
        .merge(jobs.rename(columns={"State": "job_state"}),
               on=["JobID", "WindowID"], how="left"))

# 1) Candidate pool coverage
print("val applications:", len(v))
print("job found in jobs table (same window):", round(v.job_state.notna().mean(), 4))
print("user found in users table:", round(v.user_state.notna().mean(), 4))
print("job state == user state:", round((v.user_state == v.job_state).mean(), 4))

# 2) Candidate pool size per user
cell_size = jobs.groupby(["WindowID", "State"]).size().rename("cell_size")
u = users.merge(cell_size, left_on=["WindowID", "State"],
                right_index=True, how="left")
print("\ncandidate pool size per user:")
print(u.cell_size.describe().round(0))

# 3) Popularity concentration in validation
top = v.JobID.value_counts()
print("\nshare of val applications going to top 1% of jobs:",
      round(top.head(max(1, int(len(top) * 0.01))).sum() / top.sum(), 3))

# 4) Were applied jobs open at application time?
dates = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t", quoting=3,
                    usecols=["JobID", "WindowID", "StartDate", "EndDate"],
                    low_memory=False)
for c in ["StartDate", "EndDate"]:
    dates[c] = pd.to_datetime(dates[c], errors="coerce")
dates = dates.drop_duplicates(["JobID", "WindowID"])
d = val.merge(dates, on=["JobID", "WindowID"], how="left")
d = d.dropna(subset=["StartDate", "EndDate"])
is_open = (d.ApplicationDate >= d.StartDate) & (d.ApplicationDate <= d.EndDate)
print("\nval applications with valid start/end dates:", len(d))
print("applied job was open at application time:", round(is_open.mean(), 4))