# src/rrf_hybrid.py
# Reciprocal rank fusion (k = 60) of candidate lists, top-K per user
import argparse

import pandas as pd

KEY = ["UserID", "WindowID", "JobID"]
RRF_K = 60


def rrf(frames, k):
    parts = []
    for f in frames:
        t = f[KEY + ["rank"]].copy()
        t["s"] = 1.0 / (RRF_K + t["rank"])
        parts.append(t[KEY + ["s"]])
    m = pd.concat(parts, ignore_index=True).groupby(KEY, as_index=False)["s"].sum()
    m = m.sort_values(["UserID", "s"], ascending=[True, False], kind="mergesort")
    m["rank"] = m.groupby("UserID").cumcount() + 1
    return m[m["rank"] <= k][KEY + ["rank"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--collab", required=True)
    ap.add_argument("--pop", default=None)       # optional third list
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()

    content = pd.read_csv(a.content, usecols=KEY + ["rank"])
    collab = pd.read_csv(a.collab, usecols=KEY + ["rank"])

    out2 = rrf([content, collab], a.k)
    out2.to_csv(f"{a.prefix}rrf.csv", index=False)
    print("saved", f"{a.prefix}rrf.csv", "| users:", out2.UserID.nunique())

    if a.pop:
        pop = pd.read_csv(a.pop, usecols=KEY + ["rank"])
        out3 = rrf([content, collab, pop], a.k)
        out3.to_csv(f"{a.prefix}rrf3.csv", index=False)
        print("saved", f"{a.prefix}rrf3.csv", "| users:", out3.UserID.nunique())


if __name__ == "__main__":
    main()