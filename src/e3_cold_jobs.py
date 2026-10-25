import numpy as np
import pandas as pd

from split_utils import load_apps, temporal_split

KEY = ["UserID", "WindowID", "JobID"]
train, _, _ = temporal_split(load_apps())
train_jobs = set(zip(train["JobID"], train["WindowID"]))

labels = pd.read_csv("results/recs/val_labels.csv")
labels["cold_job"] = [(j, w) not in train_jobs for j, w in zip(labels["JobID"], labels["WindowID"])]
print("val targets:", len(labels), "| share on cold jobs (no train application):",
      round(labels["cold_job"].mean(), 4))

cold = labels[labels["cold_job"]]
rel = {}
for u, w, j in zip(cold["UserID"], cold["WindowID"], cold["JobID"]):
    rel.setdefault(u, set()).add((j, w))
print("users with at least one cold-job target:", len(rel))

files = {
    "pop_state_open": "results/recs/val_pop_state_open.csv",
    "content_bge": "results/recs/val_content.csv",
    "collab_itemitem": "results/recs/val_collab.csv",
    "hybrid_rrf (content+collab)": "results/recs/val_hybrid_rrf.csv",
    "hybrid_rrf3 (+pop)": "results/recs/val_hybrid_rrf3.csv",
}

rows = []
for name, path in files.items():
    recs = pd.read_csv(path).sort_values(["UserID", "rank"])
    lists = {u: list(zip(g["JobID"], g["WindowID"])) for u, g in recs.groupby("UserID", sort=False)}
    res = {k: [] for k in (10, 50, 100)}
    for u, rset in rel.items():
        lst = lists.get(u, [])
        for k in res:
            res[k].append(len(set(lst[:k]) & rset) / len(rset))
    rows.append({"method": name, "users": len(rel),
                 **{f"ColdJobRecall@{k}": round(float(np.mean(v)), 4) for k, v in res.items()}})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("results/e3_cold_jobs.csv", index=False)