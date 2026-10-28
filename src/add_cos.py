# src/add_cos.py
# Adds a `cos` column: content cosine for EVERY (user, job) row, computed from the embeddings
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    f = pd.read_parquet(a.features)
    jobs = pd.read_parquet("artifacts/jobs_open_union.parquet", columns=["JobID", "WindowID"])
    jobs["ji"] = np.arange(len(jobs))
    jobs = jobs.drop_duplicates(["JobID", "WindowID"])
    users = pd.read_parquet("artifacts/users_keys.parquet", columns=["UserID", "WindowID"])
    users["ui"] = np.arange(len(users))
    users = users.drop_duplicates(["UserID", "WindowID"])
    J = np.load("artifacts/job_union_bge_small_title_req.npy")
    U = np.load("artifacts/user_emb_bge_small.npy")

    keys = f[["UserID", "WindowID", "JobID"]]
    ji = keys.merge(jobs, on=["JobID", "WindowID"], how="left")["ji"].to_numpy()
    ui = keys.merge(users, on=["UserID", "WindowID"], how="left")["ui"].to_numpy()

    ok = ~(np.isnan(ji) | np.isnan(ui))
    pos = np.flatnonzero(ok)
    cos = np.full(len(f), np.nan, dtype=np.float32)
    for s in range(0, len(pos), 200_000):
        b = pos[s:s + 200_000]
        jv = J[ji[b].astype(np.int64)].astype(np.float32)
        uv = U[ui[b].astype(np.int64)].astype(np.float32)
        cos[b] = (jv * uv).sum(axis=1)
    f["cos"] = cos

    print("rows:", len(f), "| missing cos:", int(np.isnan(cos).sum()))
    if "content_score" in f.columns:
        m = f["content_score"].notna() & f["cos"].notna()
        diff = (f.loc[m, "cos"] - f.loc[m, "content_score"]).abs().max()
        print("check vs content_score | rows compared:", int(m.sum()), "| max abs diff:", float(diff))
    f.to_parquet(a.out, index=False)
    print("saved:", a.out)


if __name__ == "__main__":
    main()