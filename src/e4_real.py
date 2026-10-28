# src/e4_real.py
# E4 on real data: hard vs random negatives, same positives, same non-retriever features
import argparse
import os

import lightgbm as lgb
import pandas as pd

FEATURES = ["cos", "same_city", "job_pop", "job_age_days", "days_left",
            "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]
PARAMS = dict(objective="lambdarank", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=50)
COLS = ["UserID", "WindowID", "JobID", "label"] + FEATURES


def sizes(df):
    return df.groupby("UserID", sort=False).size().values


def with_pos(d):
    return d[d.UserID.isin(d.loc[d.label == 1, "UserID"].unique())]


def srt(d):
    return d.sort_values("UserID", kind="mergesort")


def fit(train, va_p, seed):
    train = srt(train)
    m = lgb.LGBMRanker(random_state=seed, verbose=-1, **PARAMS)
    m.fit(train[FEATURES], train["label"], group=sizes(train),
          eval_set=[(va_p[FEATURES], va_p["label"])], eval_group=[sizes(va_p)],
          eval_at=[10], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    return m


def write_recs(m, va, path, k):
    t = va[["UserID", "WindowID", "JobID"]].copy()
    t["s"] = m.predict(va[FEATURES])
    t = t.sort_values(["UserID", "s"], ascending=[True, False], kind="mergesort")
    t["rank"] = t.groupby("UserID").cumcount() + 1
    t[t["rank"] <= k][["UserID", "WindowID", "JobID", "rank"]].to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", default="data/processed/features_train_rr_cos.parquet")
    ap.add_argument("--rand", default="data/processed/features_train_rr_rand_cos.parquet")
    ap.add_argument("--val", default="data/processed/features_val_cos.parquet")
    ap.add_argument("--prefix", default="results/recs/e4_")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    hard = with_pos(pd.read_parquet(a.hard, columns=COLS))
    users = set(hard.UserID.unique())
    pos = hard[hard.label == 1]
    hneg = hard[hard.label == 0]
    rneg = pd.read_parquet(a.rand, columns=COLS)
    rneg = rneg[(rneg.label == 0) & rneg.UserID.isin(users)]
    print("users:", len(users), "| positives:", len(pos),
          "| hard negatives:", len(hneg), "| random negatives:", len(rneg))

    va = srt(pd.read_parquet(a.val, columns=COLS))
    va_p = with_pos(va)

    os.makedirs(os.path.dirname(a.prefix), exist_ok=True)
    for name, train in [("hard", pd.concat([pos, hneg])), ("rand", pd.concat([pos, rneg]))]:
        m = fit(train, va_p, a.seed)
        print(name, "| rows:", len(train), "| best iteration:", m.best_iteration_)
        write_recs(m, va, f"{a.prefix}{name}.csv", a.k)
        print("saved", f"{a.prefix}{name}.csv")


if __name__ == "__main__":
    main()