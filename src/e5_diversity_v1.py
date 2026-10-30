# src/e5_diversity_v1.py
# Popularity-penalised re-scoring of the real re-ranker; writes one recs file per lambda
import argparse
import os

import numpy as np
import pandas as pd

from predict import load


def zscore(df, col):
    g = df.groupby("UserID")[col]
    sd = g.transform("std").fillna(1.0).replace(0, 1.0)
    return (df[col] - g.transform("mean")) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/reranker_v1")
    ap.add_argument("--features", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.25, 0.5])
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--prefix", default="results/recs/rerank_v1_lam")
    a = ap.parse_args()

    booster, meta = load(a.artifact)
    d = pd.read_parquet(a.features)
    d["s"] = booster.predict(d[meta["features"]])
    d["lp"] = np.log1p(d["job_pop"])
    d["zs"] = zscore(d, "s")
    d["zp"] = zscore(d, "lp")

    os.makedirs(os.path.dirname(a.prefix), exist_ok=True)
    for lam in a.lambdas:
        t = d[["UserID", "WindowID", "JobID"]].copy()
        t["adj"] = d["zs"] - lam * d["zp"]
        t = t.sort_values(["UserID", "adj"], ascending=[True, False], kind="mergesort")
        t["rank"] = t.groupby("UserID").cumcount() + 1
        out = f"{a.prefix}{lam:g}.csv"
        t[t["rank"] <= a.k][["UserID", "WindowID", "JobID", "rank"]].to_csv(out, index=False)
        print("saved", out)


if __name__ == "__main__":
    main()