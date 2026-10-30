# src/e4_negatives.py  (smoke test on the val sample: NOT for the report)
import numpy as np
import pandas as pd
import lightgbm as lgb

NR = ["same_city", "job_pop", "job_age_days", "days_left",
      "user_n_hist", "TotalYearsExperience", "WorkHistoryCount"]

hard = pd.read_parquet("data/processed/features_val_sample.parquet")
rand = pd.read_parquet("data/processed/features_val_random.parquet")
rand_neg = rand[rand.label == 0]
eval_users_all = hard.loc[hard.label == 1, "UserID"].unique()


def group_sizes(df):
    return df.groupby("UserID", sort=False).size().values


def metrics(df, score):
    d = df[["UserID", "label"]].copy()
    d["s"] = np.asarray(score)
    d = d.sort_values(["UserID", "s"], ascending=[True, False], kind="mergesort")
    d["pos"] = d.groupby("UserID").cumcount() + 1
    top = d[d.pos <= 10]
    npos = d.groupby("UserID").label.sum()
    dcg = (top.label / np.log2(top.pos + 1)).groupby(top.UserID).sum()
    dcg = dcg.reindex(npos.index).fillna(0)
    idcg = npos.map(lambda n: sum(1 / np.log2(i + 1) for i in range(1, min(int(n), 10) + 1)))
    p10 = top.groupby("UserID").label.sum().reindex(npos.index).fillna(0) / 10
    return (dcg / idcg).mean(), p10.mean()


def train(df, seed):
    df = df.sort_values("UserID", kind="mergesort")
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05,
                       num_leaves=31, random_state=seed, verbose=-1)
    m.fit(df[NR], df["label"], group=group_sizes(df))
    return m


rows = []
for seed in [42, 43, 44]:
    rng = np.random.default_rng(seed)
    users = eval_users_all.copy()
    rng.shuffle(users)
    cut = int(len(users) * 0.8)
    tr_u, te_u = set(users[:cut]), set(users[cut:])

    tr_hard = hard[hard.UserID.isin(tr_u)]
    pos = tr_hard[tr_hard.label == 1]
    tr_rand = pd.concat([pos, rand_neg[rand_neg.UserID.isin(tr_u)]], ignore_index=True)
    te = hard[hard.UserID.isin(te_u)]

    res = {
        "popularity_only": metrics(te, te.job_pop.fillna(0)),
        "content_order": metrics(te, -te.content_rank.fillna(1e9)),
        "lgbm_hard_neg": metrics(te, train(tr_hard, seed).predict(te[NR])),
        "lgbm_random_neg": metrics(te, train(tr_rand, seed).predict(te[NR])),
    }
    for k, (n, p) in res.items():
        rows.append({"seed": seed, "method": k, "NDCG@10": n, "P@10": p})
    print("seed", seed, "| eval users:", te[te.label == 1].UserID.nunique(),
          "| train rows hard/random:", len(tr_hard), len(tr_rand))

out = pd.DataFrame(rows).groupby("method")[["NDCG@10", "P@10"]].mean().round(4)
print(out.sort_values("NDCG@10", ascending=False))