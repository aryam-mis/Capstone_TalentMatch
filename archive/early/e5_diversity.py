# src/e5_diversity.py  (smoke test on the val sample: NOT for the report)
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATURES = ["content_score", "content_rank", "collab_score", "collab_rank",
            "same_city", "job_pop", "job_age_days", "days_left",
            "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]
HEAD_MIN_POP = 14   # min train-pop inside the top 1% most popular open jobs
LAMBDAS = [0, 0.05, 0.1, 0.15, 0.25, 0.5]
K = 10

d = pd.read_parquet("data/processed/features_val_sample.parquet")
rng = np.random.default_rng(42)
users = d.UserID.unique()
rng.shuffle(users)
cut = int(len(users) * 0.8)
tr_u, te_u = set(users[:cut]), set(users[cut:])
has_pos = set(d.loc[d.label == 1, "UserID"].unique())

tr = d[d.UserID.isin(tr_u & has_pos)].sort_values("UserID", kind="mergesort")
te = d[d.UserID.isin(te_u)].sort_values("UserID", kind="mergesort").reset_index(drop=True)

model = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05,
                       num_leaves=31, random_state=42, verbose=-1)
model.fit(tr[FEATURES], tr["label"], group=tr.groupby("UserID", sort=False).size().values)
te["s"] = model.predict(te[FEATURES])
te["lp"] = np.log1p(te["job_pop"])


def zscore(col):
    g = te.groupby("UserID")[col]
    sd = g.transform("std").fillna(1).replace(0, 1)
    return (te[col] - g.transform("mean")) / sd


te["zs"], te["zp"] = zscore("s"), zscore("lp")


def evaluate(lam):
    t = te[["UserID", "JobID", "WindowID", "label", "job_pop"]].copy()
    t["adj"] = te["zs"] - lam * te["zp"]
    t = t.sort_values(["UserID", "adj"], ascending=[True, False], kind="mergesort")
    t["pos"] = t.groupby("UserID").cumcount() + 1
    top = t[t.pos <= K]

    npos = t.groupby("UserID").label.sum()
    pu = npos[npos > 0].index                      # accuracy only on users with a positive
    tp = top[top.UserID.isin(pu)]
    dcg = (tp.label / np.log2(tp.pos + 1)).groupby(tp.UserID).sum().reindex(pu).fillna(0)
    idcg = npos[pu].map(lambda n: sum(1 / np.log2(i + 1) for i in range(1, min(int(n), K) + 1)))
    ndcg = (dcg / idcg).mean()
    p10 = tp.groupby("UserID").label.sum().reindex(pu).fillna(0).div(K).mean()

    # diversity on ALL held-out users
    uniq = len(top[["JobID", "WindowID"]].drop_duplicates())
    return {"lambda": lam, "NDCG@10": round(ndcg, 4), "P@10": round(p10, 4),
            "unique/recs_%": round(uniq / len(top) * 100, 1),
            "head_share_%": round((top.job_pop >= HEAD_MIN_POP).mean() * 100, 1)}


print("held-out users:", te.UserID.nunique(), "| with a positive:", te[te.label == 1].UserID.nunique())
print(pd.DataFrame([evaluate(l) for l in LAMBDAS]).to_string(index=False))