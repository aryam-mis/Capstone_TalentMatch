# src/save_artifact.py  (v0 on the val sample: pipeline check only, not a final model)
import argparse
import json
import os
import platform

import lightgbm as lgb
import pandas as pd

FEATURES = ["content_score", "content_rank", "collab_score", "collab_rank",
            "same_city", "job_pop", "job_age_days", "days_left",
            "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]
PARAMS = dict(objective="lambdarank", n_estimators=200, learning_rate=0.05, num_leaves=31)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="artifacts/reranker_v0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=0.0)  # diversity penalty weight
    a = ap.parse_args()

    d = pd.read_parquet(a.features)
    has_pos = d.loc[d.label == 1, "UserID"].unique()
    d = d[d.UserID.isin(has_pos)].sort_values("UserID", kind="mergesort")

    model = lgb.LGBMRanker(random_state=a.seed, verbose=-1, **PARAMS)
    model.fit(d[FEATURES], d["label"], group=d.groupby("UserID", sort=False).size().values)

    os.makedirs(a.out, exist_ok=True)
    model.booster_.save_model(os.path.join(a.out, "model.txt"))
    meta = {"features": FEATURES, "params": PARAMS, "seed": a.seed,
            "diversity_lambda": a.lam, "trained_on": a.features,
            "n_users": int(d.UserID.nunique()), "n_rows": len(d),
            "lightgbm": lgb.__version__, "python": platform.python_version()}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("saved to", a.out, "| users:", meta["n_users"], "| rows:", meta["n_rows"])


if __name__ == "__main__":
    main()