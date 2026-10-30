# src/compare_lr.py  (smoke test on the val sample: NOT for the report)
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = ["content_score", "content_rank", "collab_score", "collab_rank",
            "same_city", "job_pop", "job_age_days", "days_left",
            "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]

d = pd.read_parquet("data/processed/features_val_sample.parquet")
d["job_pop"] = np.log1p(d["job_pop"])
has_pos = set(d.loc[d.label == 1, "UserID"].unique())


def metrics(df, score):
    t = df[["UserID", "label"]].copy()
    t["s"] = np.asarray(score)
    t = t.sort_values(["UserID", "s"], ascending=[True, False], kind="mergesort")
    t["pos"] = t.groupby("UserID").cumcount() + 1
    npos = t.groupby("UserID").label.sum()
    pu = npos[npos > 0].index
    top = t[(t.pos <= 10) & t.UserID.isin(pu)]
    dcg = (top.label / np.log2(top.pos + 1)).groupby(top.UserID).sum().reindex(pu).fillna(0)
    idcg = npos[pu].map(lambda n: sum(1 / np.log2(i + 1) for i in range(1, min(int(n), 10) + 1)))
    p10 = top.groupby("UserID").label.sum().reindex(pu).fillna(0).div(10)
    return (dcg / idcg).mean(), p10.mean()


rows = []
for seed in [42, 43, 44]:
    rng = np.random.default_rng(seed)
    users = d.UserID.unique()
    rng.shuffle(users)
    cut = int(len(users) * 0.8)
    tr_u, te_u = set(users[:cut]), set(users[cut:])

    tr = d[d.UserID.isin(tr_u & has_pos)].sort_values("UserID", kind="mergesort")
    te = d[d.UserID.isin(te_u)]

    # LightGBM
    gb = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05,
                        num_leaves=31, random_state=seed, verbose=-1)
    gb.fit(tr[FEATURES], tr["label"], group=tr.groupby("UserID", sort=False).size().values)

    # Logistic Regression (imputer + scaler are fit on the training rows only)
    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                       StandardScaler(),
                       LogisticRegression(max_iter=1000, class_weight="balanced"))
    lr.fit(tr[FEATURES], tr["label"])

    res = {
        "popularity_only": metrics(te, te.job_pop),
        "logistic_regression": metrics(te, lr.predict_proba(te[FEATURES])[:, 1]),
        "lightgbm_ranker": metrics(te, gb.predict(te[FEATURES])),
    }
    for k, (n, p) in res.items():
        rows.append({"seed": seed, "method": k, "NDCG@10": n, "P@10": p})

out = pd.DataFrame(rows).groupby("method")[["NDCG@10", "P@10"]].mean().round(4)
print("eval users with a positive (last seed):", te[te.label == 1].UserID.nunique())
print(out.sort_values("NDCG@10", ascending=False))