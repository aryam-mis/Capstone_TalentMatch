import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

from split_utils import load_apps, temporal_split

p = argparse.ArgumentParser()
p.add_argument("--register", action="store_true",
               help="also run evaluate.py for the new methods and append to results/metrics.csv")
args = p.parse_args()

KEY = ["UserID", "WindowID", "JobID"]
K = 100
RRF_K = 60
VAL_START, VAL_END = pd.Timestamp("2012-06-06"), pd.Timestamp("2012-06-14")
BINS = [("0 (cold-start)", 0, 0), ("1-2", 1, 2), ("3+", 3, 10**9)]

# ---------- Load ----------
train, _, _ = temporal_split(load_apps())
labels = pd.read_csv("results/recs/val_labels.csv")
content = pd.read_csv("results/recs/val_content.csv")
collab = pd.read_csv("results/recs/val_collab.csv")

users = pd.read_parquet("artifacts/users_keys.parquet").drop_duplicates(["UserID", "WindowID"])
val_pairs = (labels[["UserID", "WindowID"]].drop_duplicates()
             .merge(users, on=["UserID", "WindowID"], how="left"))
assert val_pairs["UserID"].is_unique, "each val user must map to one window"

hist = train[KEY].drop_duplicates()
n_hist = hist.groupby(["UserID", "WindowID"]).size().rename("n_hist").reset_index()
val_pairs = val_pairs.merge(n_hist, on=["UserID", "WindowID"], how="left").fillna({"n_hist": 0})
group_of = dict(zip(val_pairs["UserID"], val_pairs["n_hist"]))

jobs = pd.read_parquet("artifacts/jobs_open_union.parquet",
                       columns=["JobID", "WindowID", "State", "StartDate", "EndDate"])
is_open = ((jobs["EndDate"].isna() | (jobs["EndDate"] >= VAL_START)) &
           (jobs["StartDate"].isna() | (jobs["StartDate"] < VAL_END)))
open_jobs = jobs[is_open].drop_duplicates(["JobID", "WindowID"])
print("open jobs for val:", len(open_jobs), "(expected 332650)")

# ---------- Baseline: state popularity with the same filters ----------
pop = hist.groupby(["JobID", "WindowID"]).size().rename("pop").reset_index()
oj = (open_jobs.merge(pop, on=["JobID", "WindowID"], how="left")
      .fillna({"pop": 0}).sort_values("pop", ascending=False, kind="stable"))
cell_jobs = {k: g["JobID"].values for k, g in oj.groupby(["WindowID", "State"])}
seen_by = (hist.merge(val_pairs[["UserID", "WindowID"]], on=["UserID", "WindowID"])
           .groupby(["UserID", "WindowID"])["JobID"].apply(set).to_dict())

rows = []
for uid, w, st in zip(val_pairs["UserID"], val_pairs["WindowID"], val_pairs["State"]):
    arr = cell_jobs.get((w, st))
    if arr is None:
        continue
    seen = seen_by.get((uid, w), set())
    rank = 0
    for j in arr:
        if j in seen:
            continue
        rank += 1
        rows.append((uid, w, j, rank))
        if rank == K:
            break
pop_recs = pd.DataFrame(rows, columns=KEY + ["rank"])


# ---------- Hybrid: reciprocal rank fusion ----------
def rrf(frames):
    parts = []
    for f in frames:
        t = f[KEY + ["rank"]].copy()
        t["s"] = 1.0 / (RRF_K + t["rank"])
        parts.append(t[KEY + ["s"]])
    m = pd.concat(parts).groupby(KEY, as_index=False)["s"].sum()
    m = m.sort_values(["UserID", "WindowID", "s"], ascending=[True, True, False], kind="stable")
    m["rank"] = m.groupby(["UserID", "WindowID"]).cumcount() + 1
    m = m[m["rank"] <= K].rename(columns={"s": "score"})
    return m[KEY + ["rank", "score"]]


hybrid2 = rrf([content, collab])
hybrid3 = rrf([content, collab, pop_recs])

methods = {
    "pop_state_open": pop_recs,
    "content_bge": content,
    "collab_itemitem": collab,
    "hybrid_rrf (content+collab)": hybrid2,
    "hybrid_rrf3 (+pop)": hybrid3,
}
hybrid2.to_csv("results/recs/val_hybrid_rrf.csv", index=False)
hybrid3.to_csv("results/recs/val_hybrid_rrf3.csv", index=False)
pop_recs.to_csv("results/recs/val_pop_state_open.csv", index=False)

# ---------- Metrics (same logic as evaluate.py: keyed by UserID, jobs as (JobID, WindowID)) ----------
rel = {}
for u, w, j in zip(labels["UserID"], labels["WindowID"], labels["JobID"]):
    rel.setdefault(u, set()).add((j, w))
disc = 1.0 / np.log2(np.arange(2, 12))


def per_user(recs):
    recs = recs.sort_values(["UserID", "rank"]).drop_duplicates(["UserID", "WindowID", "JobID"])
    lists = {u: list(zip(g["JobID"], g["WindowID"])) for u, g in recs.groupby("UserID", sort=False)}
    out = {}
    for u, rset in rel.items():
        lst = lists.get(u, [])
        hits = np.array([1 if x in rset else 0 for x in lst[:100]])
        h10 = hits[:10]
        n_rel = len(rset)
        m = min(n_rel, 10)
        dcg = float((h10 * disc[:len(h10)]).sum())
        idcg = float(disc[:m].sum())
        prec = np.cumsum(h10) / np.arange(1, len(h10) + 1) if len(h10) else np.array([])
        ap = float((prec * h10).sum() / m) if len(h10) else 0.0
        out[u] = {"P@10": h10.sum() / 10, "NDCG@10": dcg / idcg, "MAP@10": ap,
                  "Recall@50": hits[:50].sum() / n_rel, "Recall@100": hits[:100].sum() / n_rel}
    return pd.DataFrame.from_dict(out, orient="index")


rows = []
for name, recs in methods.items():
    pu = per_user(recs)
    pu["n_hist"] = pu.index.map(group_of)
    rows.append({"method": name, "group": "ALL", "users": len(pu), **pu.drop(columns="n_hist").mean().round(4)})
    for label, lo, hi in BINS:
        sub = pu[(pu["n_hist"] >= lo) & (pu["n_hist"] <= hi)]
        rows.append({"method": name, "group": label, "users": len(sub),
                     **sub.drop(columns="n_hist").mean().round(4)})

res = pd.DataFrame(rows)
res.to_csv("results/e3_by_history.csv", index=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False))

if args.register:
    for name, fname in [("e1_pop_state_open", "val_pop_state_open"),
                        ("e1_hybrid_rrf", "val_hybrid_rrf"),
                        ("e1_hybrid_rrf3", "val_hybrid_rrf3")]:
        subprocess.run([sys.executable, "src/evaluate.py", "--recs", f"results/recs/{fname}.csv",
                        "--split", "val", "--name", name], check=True)