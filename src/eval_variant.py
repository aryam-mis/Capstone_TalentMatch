import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import faiss

from split_utils import load_apps, temporal_split

p = argparse.ArgumentParser()
p.add_argument("--jobs_emb", required=True, help="npy of open-job embeddings")
p.add_argument("--users_emb", default="artifacts/user_emb_minilm_fp16.npy")
p.add_argument("--name", required=True, help="method name written to results/metrics.csv")
args = p.parse_args()

K_FINAL, K_RETRIEVE = 100, 150
os.makedirs("results/recs", exist_ok=True)

# Open jobs and their embeddings (same row order)
jobs = pd.read_parquet("artifacts/jobs_open_variants.parquet")[["JobID", "WindowID", "State"]]
jobs = jobs.reset_index(drop=True)
J = np.load(args.jobs_emb, mmap_mode="r")
assert len(J) == len(jobs), (len(J), len(jobs))

# Users and their embeddings (same row order)
users = pd.read_parquet("artifacts/users_keys.parquet").reset_index(drop=True)
U = np.load(args.users_emb, mmap_mode="r")
assert len(U) == len(users), (len(U), len(users))

# Validation users and seen applications
train, val, _ = temporal_split(load_apps())
val_pairs = val[["UserID", "WindowID"]].drop_duplicates()
pair_mask = pd.MultiIndex.from_frame(users[["UserID", "WindowID"]]).isin(
    pd.MultiIndex.from_frame(val_pairs))
val_users = users[pair_mask]
seen = train[["UserID", "WindowID", "JobID"]].drop_duplicates().assign(_seen=1)

# Retrieval: one FAISS index per (WindowID, State) cell
cells = jobs.groupby(["WindowID", "State"]).indices
job_ids = jobs["JobID"].values
rows = []
for (w, st), ug in val_users.groupby(["WindowID", "State"]):
    jidx = cells.get((w, st))
    if jidx is None:
        continue
    index = faiss.IndexFlatIP(J.shape[1])
    index.add(np.ascontiguousarray(J[jidx], dtype="float32"))
    q = np.ascontiguousarray(U[ug.index.to_numpy()], dtype="float32")
    scores, pos = index.search(q, min(K_RETRIEVE, len(jidx)))
    for uid, s_row, p_row in zip(ug["UserID"].values, scores, pos):
        for rank, (s, pp) in enumerate(zip(s_row, p_row), start=1):
            rows.append((uid, w, job_ids[jidx[pp]], rank, float(s)))

recs = pd.DataFrame(rows, columns=["UserID", "WindowID", "JobID", "rank", "score"])
recs = recs.merge(seen, on=["UserID", "WindowID", "JobID"], how="left")
recs = recs[recs["_seen"].isna()].drop(columns="_seen")
recs = recs.sort_values(["UserID", "WindowID", "rank"])
recs["rank"] = recs.groupby(["UserID", "WindowID"]).cumcount() + 1
recs = recs[recs["rank"] <= K_FINAL]

out = f"results/recs/{args.name}.csv"
recs[["UserID", "WindowID", "JobID", "rank", "score"]].to_csv(out, index=False)
print("saved", out, len(recs))

subprocess.run([sys.executable, "src/evaluate.py", "--recs", out,
                "--split", "val", "--name", args.name], check=True)