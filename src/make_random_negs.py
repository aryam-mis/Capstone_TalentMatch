# src/make_random_negs.py
import argparse
import numpy as np
import pandas as pd
from filters import open_jobs, VAL_START, VAL_END


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)  # to reuse exactly the same users
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    uids = pd.read_parquet(a.features, columns=["UserID"]).UserID.unique()

    users = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                        usecols=["UserID", "WindowID", "State"], dtype={"State": str})
    users = users[users.UserID.isin(uids)].dropna(subset=["State"])

    jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                       usecols=["JobID", "WindowID", "State", "StartDate", "EndDate"],
                       dtype={"State": str})
    jobs = open_jobs(jobs, VAL_START, VAL_END)

    tr = pd.read_csv("results/split/train.csv", usecols=["UserID", "WindowID", "JobID"])
    seen = tr[tr.UserID.isin(uids)].groupby("UserID")["JobID"].apply(set).to_dict()

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
    out.to_csv(a.out, index=False)
    print("users:", out.UserID.nunique(), "| rows:", len(out),
          "| avg per user:", round(len(out) / out.UserID.nunique(), 1))


if __name__ == "__main__":
    main()