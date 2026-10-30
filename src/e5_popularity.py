# src/e5_popularity.py
import argparse

import pandas as pd

from filters import open_jobs
from split_utils import load_apps

KEYS = ["JobID", "WindowID"]


def read_any(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--head-pct", type=float, default=1.0)
    ap.add_argument("--open_from", default="2012-06-06")
    ap.add_argument("--end", default="2012-06-14")
    ap.add_argument("--history_before", default="2012-06-06")
    a = ap.parse_args()

    jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                       usecols=KEYS + ["StartDate", "EndDate"])
    cat = open_jobs(jobs, a.open_from, a.end)[KEYS]

    # popularity from applications before the cutoff only
    apps = load_apps()
    hist = apps[apps.ApplicationDate < pd.Timestamp(a.history_before)]
    pop = hist.groupby(KEYS).size().rename("pop").reset_index()
    cat = cat.merge(pop, on=KEYS, how="left").fillna({"pop": 0})
    cat = cat.sort_values("pop", ascending=False, kind="mergesort")

    n_head = int(len(cat) * a.head_pct / 100)
    head = cat.head(n_head)[KEYS].assign(is_head=1)
    print("open catalogue:", len(cat), "| head size:", n_head,
          "| min pop in head:", int(cat.head(n_head)["pop"].min()))

    rows = []
    for path in a.recs:
        r = read_any(path)
        r = r[r["rank"] <= a.k][["UserID"] + KEYS]
        r = r.merge(head, on=KEYS, how="left")
        rows.append({
            "file": path.split("\\")[-1],
            "users": r.UserID.nunique(),
            "recs": len(r),
            "unique_jobs": len(r[KEYS].drop_duplicates()),
            "coverage_%": round(len(r[KEYS].drop_duplicates()) / len(cat) * 100, 2),
            f"head{a.head_pct:g}%_share_%": round(r["is_head"].notna().mean() * 100, 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()