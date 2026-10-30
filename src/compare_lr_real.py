# src/compare_lr_real.py
# Logistic Regression baseline for the re-ranker: same train data and features as the LightGBM model
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = ["content_score", "content_rank", "collab_score", "collab_rank",
        "same_city", "job_pop", "job_age_days", "days_left",
        "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]


def prep(d, feats):
    x = d[feats].copy()
    x["job_pop"] = np.log1p(x["job_pop"])
    return x


def write_recs(lr, path_in, path_out, feats, k):
    d = pd.read_parquet(path_in)
    t = d[["UserID", "WindowID", "JobID"]].copy()
    t["s"] = lr.predict_proba(prep(d, feats))[:, 1]
    t = t.sort_values(["UserID", "s"], ascending=[True, False], kind="mergesort")
    t["rank"] = t.groupby("UserID").cumcount() + 1
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    t[t["rank"] <= k][["UserID", "WindowID", "JobID", "rank"]].to_csv(path_out, index=False)
    print("recs saved:", path_out, "| users:", t.UserID.nunique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/features_train_rr.parquet")
    ap.add_argument("--val", default="data/processed/features_val.parquet")
    ap.add_argument("--recs", default="results/recs/lr_v1.csv")
    ap.add_argument("--test", default=None)
    ap.add_argument("--test_recs", default=None)
    ap.add_argument("--with_pop", action="store_true")   # add pop_rank (the v2 pool)
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    feats = BASE + (["pop_rank"] if a.with_pop else [])

    tr = pd.read_parquet(a.train)
    tr = tr[tr.UserID.isin(tr.loc[tr.label == 1, "UserID"].unique())]
    print("train users/rows:", tr.UserID.nunique(), len(tr), "| features:", len(feats))

    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                       StandardScaler(),
                       LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
    lr.fit(prep(tr, feats), tr["label"])
    del tr

    write_recs(lr, a.val, a.recs, feats, a.k)
    if a.test:
        write_recs(lr, a.test, a.test_recs, feats, a.k)


if __name__ == "__main__":
    main()