# src/build_features.py
import argparse

import numpy as np
import pandas as pd

from split_utils import load_apps

KEYS = ["UserID", "WindowID", "JobID"]


def read_any(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--collab", required=True)
    ap.add_argument("--pop", default=None)           # optional third candidate source
    ap.add_argument("--labels", required=True)       # target-period applications
    ap.add_argument("--pop-before", required=True)   # popularity/history use applications before this date
    ap.add_argument("--period-start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-users", type=int, default=0)
    a = ap.parse_args()

    pop_before = pd.Timestamp(a.pop_before)
    period_start = pd.Timestamp(a.period_start)

    # 1) union of candidates from all sources
    ct = read_any(a.content)[KEYS + ["rank", "score"]]
    ct = ct.rename(columns={"rank": "content_rank", "score": "content_score"})
    cb = read_any(a.collab)[KEYS + ["rank", "score"]]
    cb = cb.rename(columns={"rank": "collab_rank", "score": "collab_score"})
    cand = ct.merge(cb, on=KEYS, how="outer")
    del ct, cb
    if a.pop:
        pp = read_any(a.pop)[KEYS + ["rank"]].rename(columns={"rank": "pop_rank"})
        cand = cand.merge(pp, on=KEYS, how="outer")
        del pp
    else:
        cand["pop_rank"] = np.nan

    if a.sample_users:
        rng = np.random.default_rng(42)
        pick = rng.choice(cand.UserID.unique(), a.sample_users, replace=False)
        cand = cand[cand.UserID.isin(pick)]
    print("candidate rows:", len(cand), "| users:", cand.UserID.nunique())

    # 2) job features
    jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                       usecols=["JobID", "WindowID", "City", "State", "StartDate", "EndDate"],
                       dtype={"City": str, "State": str})
    jobs["StartDate"] = pd.to_datetime(jobs["StartDate"], errors="coerce")
    jobs["EndDate"] = pd.to_datetime(jobs["EndDate"], errors="coerce")
    jobs = jobs.rename(columns={"City": "j_city", "State": "j_state"})
    cand = cand.merge(jobs, on=["JobID", "WindowID"], how="left")
    del jobs

    # 3) user features
    users = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                        usecols=["UserID", "WindowID", "City", "State",
                                 "TotalYearsExperience", "WorkHistoryCount"],
                        dtype={"City": str, "State": str})
    users = users.rename(columns={"City": "u_city", "State": "u_state"})
    cand = cand.merge(users, on=["UserID", "WindowID"], how="left")
    del users

    cand["same_state"] = (cand.j_state.str.strip().str.upper() ==
                          cand.u_state.str.strip().str.upper()).astype("int8")
    cand["same_city"] = (cand.j_city.str.strip().str.upper() ==
                         cand.u_city.str.strip().str.upper()).astype("int8")
    cand["job_age_days"] = (period_start - cand.StartDate).dt.days
    cand["days_left"] = (cand.EndDate - period_start).dt.days

    # 4) popularity + user history: applications BEFORE the cutoff only (from all applications)
    apps = load_apps()
    hist = apps[apps.ApplicationDate < pop_before]
    print("history rows before", a.pop_before, ":", len(hist))

    pop = hist.groupby(["JobID", "WindowID"]).size().rename("job_pop").reset_index()
    cand = cand.merge(pop, on=["JobID", "WindowID"], how="left")
    cand["job_pop"] = cand["job_pop"].fillna(0)

    nh = hist.groupby("UserID").size().rename("user_n_hist").reset_index()
    cand = cand.merge(nh, on="UserID", how="left")
    cand["user_n_hist"] = cand["user_n_hist"].fillna(0)

    # 5) labels: did the user apply in the target period?
    lab = pd.read_csv(a.labels, usecols=KEYS).drop_duplicates().assign(label=1)
    cand = cand.merge(lab, on=KEYS, how="left")
    cand["label"] = cand["label"].fillna(0).astype("int8")

    # 6) keep needed columns, sort so each user's rows are contiguous (LightGBM groups)
    cols = KEYS + ["label", "content_score", "content_rank", "collab_score", "collab_rank",
                   "pop_rank", "same_state", "same_city", "job_pop", "job_age_days",
                   "days_left", "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]
    cand = cand[cols].sort_values(["UserID", "WindowID"]).reset_index(drop=True)
    for c in cols[4:]:
        cand[c] = cand[c].astype("float32")

    cand.to_parquet(a.out, index=False)
    print("saved:", a.out)
    print("rows:", len(cand), "| users:", cand.UserID.nunique(),
          "| positives:", int(cand.label.sum()),
          "| users with >=1 positive:", cand[cand.label == 1].UserID.nunique())


if __name__ == "__main__":
    main()