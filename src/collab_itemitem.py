import os
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from split_utils import load_apps, temporal_split

K_FINAL = 100
CHUNK = 500
NAME = "collab_itemitem"
os.makedirs("results/recs", exist_ok=True)

train, val, _ = temporal_split(load_apps())
open_jobs = (pd.read_parquet("artifacts/jobs_open_variants.parquet")
               [["JobID", "WindowID", "State"]]
               .drop_duplicates(["JobID", "WindowID"]))
users = (pd.read_parquet("artifacts/users_keys.parquet")
           .drop_duplicates(["UserID", "WindowID"]))

# ---------- User-item matrix from TRAIN only ----------
tr = train[["UserID", "WindowID", "JobID"]].drop_duplicates()
u_codes, u_uniques = pd.factorize(pd.MultiIndex.from_frame(tr[["UserID", "WindowID"]]))
i_codes, i_uniques = pd.factorize(pd.MultiIndex.from_frame(tr[["JobID", "WindowID"]]))
user_lookup = {k: i for i, k in enumerate(u_uniques)}

R = sp.csr_matrix((np.ones(len(tr), dtype=np.float32), (u_codes, i_codes)),
                  shape=(len(u_uniques), len(i_uniques)))
pop = np.asarray(R.sum(axis=0)).ravel()
Rn = (R @ sp.diags(1.0 / np.sqrt(pop))).tocsr()      # columns scaled by 1/sqrt(popularity)
RnT = Rn.T.tocsr()
print("train users:", R.shape[0], "| train items:", R.shape[1], "| interactions:", R.nnz)

# ---------- Item metadata (open flag, window, state) ----------
item_df = pd.DataFrame(list(i_uniques), columns=["JobID", "WindowID"]).merge(
    open_jobs, on=["JobID", "WindowID"], how="left")
item_job = item_df["JobID"].values
item_window = item_df["WindowID"].values
item_open = item_df["State"].notna().values
item_state = item_df["State"].fillna("").values

# ---------- Validation users that have train history ----------
vp = (val[["UserID", "WindowID"]].drop_duplicates()
        .merge(users, on=["UserID", "WindowID"], how="left"))
cands = [(u, w, st, user_lookup[(u, w)])
         for u, w, st in zip(vp.UserID, vp.WindowID, vp.State)
         if (u, w) in user_lookup]
print("val pairs:", len(vp), "| with train history:", len(cands))

# ---------- Score: (history @ Rn^T) @ Rn, then filter candidates ----------
rows = []
for s in range(0, len(cands), CHUNK):
    part = cands[s:s + CHUNK]
    H = R[[c[3] for c in part]]
    scores = ((H @ RnT) @ Rn).tocsr()
    for r, (u, w, st, uc) in enumerate(part):
        a, b = scores.indptr[r], scores.indptr[r + 1]
        idx, sc = scores.indices[a:b], scores.data[a:b]
        keep = item_open[idx] & (item_window[idx] == w) & (item_state[idx] == st)
        keep &= ~np.isin(idx, R.indices[R.indptr[uc]:R.indptr[uc + 1]])   # drop already-applied jobs
        idx, sc = idx[keep], sc[keep]
        if len(idx) == 0:
            continue
        top = np.argsort(-sc)[:K_FINAL]
        for rank, t in enumerate(top, start=1):
            rows.append((u, w, item_job[idx[t]], rank, float(sc[t])))
    print("processed", min(s + CHUNK, len(cands)), "/", len(cands), flush=True)

recs = pd.DataFrame(rows, columns=["UserID", "WindowID", "JobID", "rank", "score"])
print("users with recs:", recs[["UserID", "WindowID"]].drop_duplicates().shape[0])

out = f"results/recs/{NAME}.csv"
recs[["UserID", "WindowID", "JobID", "rank"]].to_csv(out, index=False)
recs.to_parquet(f"results/recs/{NAME}_scores.parquet", index=False)   # scores kept for the hybrid
print("saved", out, len(recs))

subprocess.run([sys.executable, "src/evaluate.py", "--recs", out,
                "--split", "val", "--name", NAME], check=True)