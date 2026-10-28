# src/train_reranker_v2.py
# Re-ranker with the popularity candidate source (pop_rank): train on train_rr, early-stop on val
import argparse
import json
import os
import platform

import lightgbm as lgb
import pandas as pd

FEATURES = ["content_score", "content_rank", "collab_score", "collab_rank", "pop_rank",
            "same_city", "job_pop", "job_age_days", "days_left",
            "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]
PARAMS = dict(objective="lambdarank", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=50)


def group_sizes(df):
    return df.groupby("UserID", sort=False).size().values


def users_with_positive(d):
    return d[d.UserID.isin(d.loc[d.label == 1, "UserID"].unique())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="artifacts/reranker_v2")
    ap.add_argument("--recs", default="results/recs/rerank_v2.csv")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tr = pd.read_parquet(a.train)
    va = pd.read_parquet(a.val)
    tr_p, va_p = users_with_positive(tr), users_with_positive(va)
    print("train users/rows:", tr_p.UserID.nunique(), len(tr_p),
          "| val users/rows (early stopping):", va_p.UserID.nunique(), len(va_p))

    model = lgb.LGBMRanker(random_state=a.seed, verbose=-1, **PARAMS)
    model.fit(tr_p[FEATURES], tr_p["label"], group=group_sizes(tr_p),
              eval_set=[(va_p[FEATURES], va_p["label"])], eval_group=[group_sizes(va_p)],
              eval_at=[10], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(25)])
    print("best iteration:", model.best_iteration_)

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    print("\nfeature importance (gain, %):")
    print((imp / imp.sum() * 100).round(1).sort_values(ascending=False).to_string())

    os.makedirs(a.out, exist_ok=True)
    model.booster_.save_model(os.path.join(a.out, "model.txt"))
    meta = {"features": FEATURES, "params": PARAMS, "seed": a.seed,
            "best_iteration": int(model.best_iteration_),
            "trained_on": a.train, "early_stopped_on": a.val,
            "lightgbm": lgb.__version__, "python": platform.python_version()}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    va = va.copy()
    va["score"] = model.predict(va[FEATURES])
    va = va.sort_values(["UserID", "score"], ascending=[True, False], kind="mergesort")
    va["rank"] = va.groupby("UserID").cumcount() + 1
    recs = va[va["rank"] <= a.k][["UserID", "WindowID", "JobID", "rank"]]
    os.makedirs(os.path.dirname(a.recs), exist_ok=True)
    recs.to_csv(a.recs, index=False)
    print("\nrecs saved:", a.recs, "| users:", recs.UserID.nunique(), "| rows:", len(recs))


if __name__ == "__main__":
    main()