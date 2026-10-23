# src/train_reranker.py  (smoke test: val sample only, NOT for the report)
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATURES = ["content_score", "content_rank", "collab_score", "collab_rank",
            "same_city", "job_pop", "job_age_days", "days_left",
            "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]
# same_state is constant in these candidates, so it is excluded


def group_sizes(df):
    # rows are sorted by UserID, so each user's rows are contiguous
    return df.groupby("UserID", sort=False).size().values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    d = pd.read_parquet(a.features)

    # split BY USER (never by row)
    rng = np.random.default_rng(a.seed)
    users = d.UserID.unique()
    rng.shuffle(users)
    cut = int(len(users) * 0.8)
    tr_users, va_users = set(users[:cut]), set(users[cut:])

    # keep only users with >=1 positive: groups with no positives carry no
    # signal, and LightGBM scores their NDCG as 1, which inflates the metric
    has_pos = set(d.loc[d.label == 1, "UserID"].unique())
    tr = d[d.UserID.isin(tr_users & has_pos)]
    va = d[d.UserID.isin(va_users & has_pos)]
    print("train users:", tr.UserID.nunique(), "| valid users:", va.UserID.nunique())

    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=300,
                            learning_rate=0.05, num_leaves=31,
                            random_state=a.seed, verbose=-1)
    ranker.fit(tr[FEATURES], tr["label"], group=group_sizes(tr),
               eval_set=[(va[FEATURES], va["label"])], eval_group=[group_sizes(va)],
               eval_at=[10],
               callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])

    imp = pd.Series(ranker.booster_.feature_importance("gain"), index=FEATURES)
    print("\nfeature importance (gain, %):")
    print((imp / imp.sum() * 100).round(1).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()