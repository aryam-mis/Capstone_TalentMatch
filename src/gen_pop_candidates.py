# src/gen_pop_candidates.py
# State-popularity candidates for any period (third candidate source for the re-ranker)
import argparse

import pandas as pd

from filters import open_jobs
from split_utils import load_apps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users_from", required=True)   # same users as the content candidates
    ap.add_argument("--open_from", required=True)    # jobs must end on/after this date
    ap.add_argument("--end", required=True)          # period end, exclusive
    ap.add_argument("--history_before", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cut = pd.Timestamp(a.history_before)

    apps = load_apps()
    hist = apps[apps.ApplicationDate < cut][["UserID", "WindowID", "JobID"]]
    print("history rows:", len(hist))

    uw = pd.read_csv(a.users_from, usecols=["UserID", "WindowID"]).drop_duplicates()
    users = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                        usecols=["UserID", "WindowID", "State"], dtype={"State": str})
    users = uw.merge(users, on=["UserID", "WindowID"], how="left").dropna(subset=["State"])

    jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                       usecols=["JobID", "WindowID", "State", "StartDate", "EndDate"],
                       dtype={"State": str})
    jobs = open_jobs(jobs, a.open_from, a.end)[["JobID", "WindowID", "State"]]
    print("open jobs:", len(jobs))

    pop = hist.groupby(["JobID", "WindowID"]).size().rename("pop").reset_index()
    pop = pop.merge(jobs, on=["JobID", "WindowID"], how="inner")
    pop = pop.sort_values(["WindowID", "State", "pop", "JobID"],
                          ascending=[True, True, False, True])
    pop["r"] = pop.groupby(["WindowID", "State"]).cumcount() + 1
    pop = pop[pop.r <= a.k + 100][["WindowID", "State", "JobID", "r", "pop"]]

    recs = users.merge(pop, on=["WindowID", "State"], how="inner")
    seen = hist[["UserID", "JobID"]].drop_duplicates().assign(seen=1)
    recs = recs.merge(seen, on=["UserID", "JobID"], how="left")
    recs = recs[recs.seen.isna()].drop(columns="seen")
    recs = recs.sort_values(["UserID", "r"])
    recs["rank"] = recs.groupby("UserID").cumcount() + 1
    recs = recs[recs["rank"] <= a.k].rename(columns={"pop": "score"})
    recs = recs[["UserID", "WindowID", "JobID", "rank", "score"]]
    recs.to_csv(a.out, index=False)
    print("saved", a.out, "| users:", recs.UserID.nunique(), "| rows:", len(recs))


if __name__ == "__main__":
    main()