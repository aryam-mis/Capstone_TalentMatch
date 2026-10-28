# src/make_random_negs.py
# Random open-job negatives for any period (same window and state as the user)
import argparse

import numpy as np
import pandas as pd

from filters import open_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)       # source of the users
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2012-06-06")   # period start (open-jobs filter)
    ap.add_argument("--end", default="2012-06-14")     # period end, exclusive
    ap.add_argument("--history_before", default=None)  # exclude jobs applied before this date
    ap.add_argument("--positive_only", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    hist_before = pd.Timestamp(a.history_before or a.start)

    f = pd.read_parquet(a.features, columns=["UserID", "label"])
    uids = f.loc[f.label == 1, "UserID"].unique() if a.positive_only else f.UserID.unique()

    users = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                        usecols=["UserID", "WindowID", "State"], dtype={"State": str})
    users = users[users.UserID.isin(uids)].dropna(subset=["State"])

    jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                       usecols=["JobID", "WindowID", "State", "StartDate", "EndDate"],
                       dtype={"State": str})
    jobs = open_jobs(jobs, a.start, a.end)

    tr = pd.read_csv("results/split/train.csv",
                     usecols=["UserID", "WindowID", "JobID", "ApplicationDate"])
    tr["ApplicationDate"] = pd.to_datetime(tr["ApplicationDate"], errors="coerce")
    tr = tr[(tr.ApplicationDate < hist_before) & tr.UserID.isin(uids)]
    seen = tr.groupby("UserID")["JobID"].apply(set).to_dict()

    pools = {k: g.JobID.values for k, g in jobs.groupby(["WindowID", "State"])}

    rows = []
    for (w, s), ug in users.groupby(["WindowID", "State"]):
        pool = pools.get((w, s))
        if pool is None:
            continue
        for uid in ug.UserID.values:
            pick = rng.choice(pool, size=min(a.n, len(pool)), replace=False)
            sn = seen.get(uid, ())
            pick = [j for j in pick if j not in sn]
            rows.append(pd.DataFrame({"UserID": uid, "WindowID": w, "JobID": pick}))

    out = pd.concat(rows, ignore_index=True)
    out["rank"] = np.nan    # build_features expects these columns
    out["score"] = np.nan
    out.to_csv(a.out, index=False)
    print("users:", out.UserID.nunique(), "| rows:", len(out),
          "| avg per user:", round(len(out) / out.UserID.nunique(), 1))


if __name__ == "__main__":
    main()