# src/judgement_metrics.py
# Mean relevance, Precision@10 and NDCG@10 on the 30-CV judgement set, for the two retrieval configurations
# and for the re-ranker, per reviewer and averaged. Unjudged re-ranker items are handled in two ways.
import argparse
import os

import numpy as np
import pandas as pd

K = 10
ap = argparse.ArgumentParser()
ap.add_argument("--comparison", required=True, help="talentmatch_reviewer_comparison.csv (452 judged items)")
ap.add_argument("--mapping", required=True, help="matched skills csv: configuration, cv_id, item_id, rank, JobID, WindowID")
ap.add_argument("--reranker", default="results/judgement/reranker_top10.csv")
ap.add_argument("--new_sheet", default=None, help="returned grading sheet for the new re-ranker items")
ap.add_argument("--new_key", default="results/judgement/reranker_new_items_key.csv")
ap.add_argument("--out", default="results/judgement/judgement_metrics.csv")
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

KEY = ["cv_id", "JobID", "WindowID"]
comp = pd.read_csv(a.comparison)
mp = pd.read_csv(a.mapping)
ids = mp[["item_id"] + KEY].drop_duplicates()
grades = comp[["item_id", "reviewer1_score", "reviewer2_score"]].merge(ids, on="item_id")
if a.new_sheet:
    sheet = pd.read_csv(a.new_sheet)[["item_id", "reviewer1_score", "reviewer2_score"]]
    sheet = sheet.dropna(subset=["reviewer1_score", "reviewer2_score"])
    new = sheet.merge(pd.read_csv(a.new_key)[["item_id"] + KEY], on="item_id")
    grades = pd.concat([grades, new], ignore_index=True)
print("judged items:", len(grades))

configs = {}
for name, g in mp.groupby("configuration"):
    configs[name] = g[KEY + ["rank"]]
if os.path.exists(a.reranker):
    configs["reranker"] = pd.read_csv(a.reranker)[KEY + ["rank"]]

cvs = sorted(comp["cv_id"].unique())
DISC = 1.0 / np.log2(np.arange(2, K + 2))


def per_cv(cfg, reviewer, fill):
    """One row per CV: mean relevance, share of 2, P@10 (score>=1), strict P@10 (score=2), NDCG@10, judged share."""
    col = f"reviewer{reviewer}_score"
    lab = grades[KEY + [col]].rename(columns={col: "s"})
    rows = []
    for cv in cvs:
        top = cfg[cfg.cv_id == cv].sort_values("rank").head(K).merge(lab, on=KEY, how="left")
        s = top["s"].values.astype(float)
        judged = ~np.isnan(s)
        if fill == "zero":
            v = np.where(judged, s, 0.0)
        else:
            v = s[judged]
        pool = np.sort(lab[lab.cv_id == cv]["s"].dropna().values)[::-1][:K]
        idcg = float((pool * DISC[:len(pool)]).sum())
        gains = np.where(judged, s, 0.0)
        dcg = float((gains[:K] * DISC[:len(gains[:K])]).sum())
        n = len(v) if len(v) else np.nan
        rows.append({"mean_rel": v.mean() if len(v) else np.nan,
                     "share_2": (v == 2).mean() if len(v) else np.nan,
                     "P10_ge1": (v >= 1).sum() / (K if fill == "zero" else n),
                     "P10_eq2": (v == 2).sum() / (K if fill == "zero" else n),
                     "NDCG10": dcg / idcg if idcg > 0 else 0.0,
                     "judged_share": judged.mean() if len(judged) else 0.0})
    return pd.DataFrame(rows, index=cvs)


rng = np.random.default_rng(a.seed)
out = []
for name, cfg in configs.items():
    for fill in ("zero", "judged_only"):
        tabs = [per_cv(cfg, r, fill) for r in (1, 2)]
        avg = (tabs[0] + tabs[1]) / 2          # average of the two reviewers, per CV
        res = {"configuration": name, "unjudged_as": fill if fill == "zero" else "ignored",
               "judged_share": round(float(avg["judged_share"].mean()), 3)}
        for m in ["mean_rel", "share_2", "P10_ge1", "P10_eq2", "NDCG10"]:
            v = avg[m].values
            v = v[~np.isnan(v)]
            boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(a.n_boot)]
            res[m] = round(float(v.mean()), 4)
            res[m + "_ci"] = f"[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]"
        for r, t in zip((1, 2), tabs):
            res[f"mean_rel_r{r}"] = round(float(t["mean_rel"].mean()), 4)
        out.append(res)

res = pd.DataFrame(out)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
res.to_csv(a.out, index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(res[["configuration", "unjudged_as", "judged_share", "mean_rel", "P10_ge1", "P10_eq2", "NDCG10"]].to_string(index=False))
print("saved", a.out)
