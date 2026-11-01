# src/baseline_popular_state_open.py
# Popularity baseline on validation: most applied-to open jobs in the user's window and state
import pandas as pd
from filters import open_jobs, VAL_START, VAL_END

K = 100        # same candidate count as the retrievers
BUFFER = 150   # take extra so at least K remain after removing jobs seen in train

tr = pd.read_csv("results/split/train.csv", usecols=["UserID", "WindowID", "JobID"])
val_users = pd.read_csv("results/split/val.csv", usecols=["UserID"]).drop_duplicates()

jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                   usecols=["JobID", "WindowID", "State", "StartDate", "EndDate"],
                   dtype={"State": str})
users = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                    usecols=["UserID", "WindowID", "State"], dtype={"State": str})

# 1) open jobs only (filtered before ranking)
jobs_open = open_jobs(jobs, VAL_START, VAL_END)[["JobID", "WindowID", "State"]]

# job popularity from train applications only
pop = tr.groupby(["JobID", "WindowID"]).size().rename("n").reset_index()
pop = pop.merge(jobs_open, on=["JobID", "WindowID"], how="inner")

# top BUFFER jobs per (window, state)
pop = pop.sort_values(["WindowID", "State", "n"], ascending=[True, True, False])
pop["pop_rank"] = pop.groupby(["WindowID", "State"]).cumcount() + 1
pop = pop[pop["pop_rank"] <= BUFFER][["WindowID", "State", "JobID", "pop_rank"]]

# attach each user's window and state
vu = val_users.merge(users, on="UserID", how="left")
print("val users:", len(vu), "| without state/window info:", vu.State.isna().sum())
vu = vu.dropna(subset=["State"])

recs = vu.merge(pop, on=["WindowID", "State"], how="inner")

# 2) remove jobs the user already applied to in train
seen = tr[["UserID", "WindowID", "JobID"]].drop_duplicates().assign(seen=1)
recs = recs.merge(seen, on=["UserID", "WindowID", "JobID"], how="left")
recs = recs[recs["seen"].isna()].drop(columns="seen")

# 3) rank and keep the top K per user
recs = recs.sort_values(["UserID", "pop_rank"])
recs["rank"] = recs.groupby("UserID").cumcount() + 1
recs = recs[recs["rank"] <= K][["UserID", "WindowID", "JobID", "rank"]]

recs.to_csv("results/recs_popular_state_open.csv", index=False)
print("users with recs:", recs.UserID.nunique(), "| rows:", len(recs),
      "| avg per user:", round(len(recs) / recs.UserID.nunique(), 1))