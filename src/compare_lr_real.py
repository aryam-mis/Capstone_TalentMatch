# src/compare_lr_real.py
# Logistic Regression baseline for the re-ranker: same train data and features as rerank_v1
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["content_score", "content_rank", "collab_score", "collab_rank",
            "same_city", "job_pop", "job_age_days", "days_left",
            "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]


def prep(d):
    x = d[FEATURES].copy()
    x["job_pop"] = np.log1p(x["job_pop"])
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/features_train_rr.parquet")
    ap.add_argument("--val", default="data/processed/features_val.parquet")
    ap.add_argument("--recs", default="results/recs/lr_v1.csv")
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()

    tr = pd.read_parquet(a.train)
    tr = tr[tr.UserID.isin(tr.loc[tr.label == 1, "UserID"].unique())]
    print("train users/rows:", tr.UserID.nunique(), len(tr))

    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                       StandardScaler(),
                       LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
    lr.fit(prep(tr), tr["label"])
    del tr

    va = pd.read_parquet(a.val)
    t = va[["UserID", "WindowID", "JobID"]].copy()
    t["s"] = lr.predict_proba(prep(va))[:, 1]
    t = t.sort_values(["UserID", "s"], ascending=[True, False], kind="mergesort")
    t["rank"] = t.groupby("UserID").cumcount() + 1
    os.makedirs(os.path.dirname(a.recs), exist_ok=True)
    t[t["rank"] <= a.k][["UserID", "WindowID", "JobID", "rank"]].to_csv(a.recs, index=False)
    print("recs saved:", a.recs, "| users:", t.UserID.nunique())


if __name__ == "__main__":
    main()