import argparse

import numpy as np
import pandas as pd

from split_utils import load_apps
from recruiter_candidates import KEY, load_assets, open_mask, finalize, evaluate

p = argparse.ArgumentParser()
p.add_argument("--period_start", default="2012-06-06")
p.add_argument("--period_end", default="2012-06-14")
p.add_argument("--labels", default="results/recs/val_labels.csv")
p.add_argument("--k", type=int, default=20)
p.add_argument("--name", default="recruiter_baseline_active")
args = p.parse_args()

start, end = pd.Timestamp(args.period_start), pd.Timestamp(args.period_end)
jobs, J, users, U = load_assets()
mask = open_mask(jobs, start, end)

# Activity = number of distinct train applications of the user (before the period start)
apps = load_apps()
hist = apps[apps["ApplicationDate"] < start][KEY].drop_duplicates()
activity = hist.groupby(["UserID", "WindowID"]).size().rename("act").reset_index()
users = users.merge(activity, on=["UserID", "WindowID"], how="left").fillna({"act": 0})

# Evaluate on the same jobs as the content retriever: open jobs with at least one real applicant
labels = pd.read_csv(args.labels)
lab_jobs = labels[["JobID", "WindowID"]].drop_duplicates()
sub = (jobs[mask][["JobID", "WindowID", "State"]].drop_duplicates(["JobID", "WindowID"])
       .merge(lab_jobs, on=["JobID", "WindowID"]))
print("open jobs with at least one real applicant:", len(sub))

# Most active users of each (window, state) cell; keep a margin for the seen-exclusion
margin = args.k + 40
top_users = {}
for key, g in users.groupby(["WindowID", "State"]):
    top_users[key] = g.sort_values("act", ascending=False, kind="stable").head(margin)["UserID"].values

parts = []
for (w, st), g in sub.groupby(["WindowID", "State"]):
    uids = top_users.get((w, st))
    if uids is None:
        continue
    n = len(uids)
    parts.append(pd.DataFrame({
        "JobID": np.repeat(g["JobID"].values, n),
        "WindowID": w,
        "UserID": np.tile(uids, len(g)),
        "score": 0.0,
        "rank": np.tile(np.arange(1, n + 1, dtype=np.int32), len(g)),
    }))
recs = pd.concat(parts, ignore_index=True)

# Same exclusion as the content retriever: users who already applied to the job before the period
recs = recs.merge(hist.assign(_seen=1), on=KEY, how="left")
recs = recs[recs["_seen"].isna()].drop(columns="_seen")
recs = finalize(recs, args.k)

out = f"results/recs/{args.name}.csv"
recs.to_csv(out, index=False)
print("saved", out, len(recs))
evaluate(recs, labels, jobs, users, ks=[10, args.k])