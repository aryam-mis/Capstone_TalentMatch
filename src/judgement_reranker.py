# src/judgement_reranker.py
# Re-ranker recommendations for the 30 judgement-set CVs, on the same validation-period jobs as recommend_cv.py
import argparse
import os
import re

import numpy as np
import pandas as pd

from profile_recommender import ProfileRecommender, POOL_CONTENT

LEVEL_YEARS = {"mid-level": 5, "senior-level": 10}   # assumption, to be stated in the report
PERIOD = dict(open_from="2012-06-06", end="2012-06-14", history_before="2012-06-06")

ap = argparse.ArgumentParser()
ap.add_argument("--comparison", required=True, help="talentmatch_reviewer_comparison.csv")
ap.add_argument("--items", required=True, help="matched skills csv (maps item_id to JobID and WindowID)")
ap.add_argument("--out", default="results/judgement/reranker_top10.csv")
ap.add_argument("--k", type=int, default=10)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)

comp = pd.read_csv(a.comparison)
cvs = comp.drop_duplicates("cv_id")[["cv_id", "cv_profile", "job_state"]].sort_values("cv_id")
items = pd.read_csv(a.items)[["cv_id", "JobID", "WindowID", "item_id"]].drop_duplicates()

rec = ProfileRecommender(period=PERIOD)
rows = []
for cv_id, profile, state in zip(cvs["cv_id"], cvs["cv_profile"], cvs["job_state"]):
    text = profile.lower()
    level = re.search(r"experience level:\s*([^.]*)\.", text).group(1).strip()
    years = float(LEVEL_YEARS.get(level, 5))
    q = rec.model.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
    parts = []
    for window in rec.windows_for(state):
        ix, emb, pop_ix = rec._pool(window, state)
        cos = emb @ q
        top = np.argsort(-cos)[:POOL_CONTENT]
        content = pd.DataFrame({"ix": ix[top], "content_score": cos[top],
                                "content_rank": np.arange(1, len(top) + 1)})
        pop = pd.DataFrame({"ix": pop_ix, "pop_rank": np.arange(1, len(pop_ix) + 1)})
        cand = content.merge(pop, on="ix", how="outer")
        j = rec.jobs.iloc[cand["ix"].values]
        f = pd.DataFrame({
            "content_score": cand["content_score"].values, "content_rank": cand["content_rank"].values,
            "collab_score": np.nan, "collab_rank": np.nan, "pop_rank": cand["pop_rank"].values,
            "same_city": 0.0, "job_pop": j.job_pop.values,
            "job_age_days": (rec.period_start - j.StartDate).dt.days.values,
            "days_left": (j.EndDate - rec.period_start).dt.days.values,
            "user_n_hist": 0.0, "TotalYearsExperience": years, "WorkHistoryCount": 1.0})
        part = j[["Title", "City", "State", "JobID", "WindowID", "job_pop"]].copy()
        part["score"] = rec.booster.predict(f[rec.meta["features"]].astype("float32"))
        part["content_cos"] = cand["content_score"].values
        parts.append(part)
    best = pd.concat(parts).sort_values("score", ascending=False).head(a.k).reset_index(drop=True)
    best.insert(0, "rank", range(1, len(best) + 1))
    best.insert(0, "cv_id", cv_id)
    rows.append(best)

out = pd.concat(rows, ignore_index=True)
out = out.merge(items, on=["cv_id", "JobID", "WindowID"], how="left")
out["already_graded"] = out["item_id"].notna()
out.to_csv(a.out, index=False, encoding="utf-8-sig")
print("saved", a.out, len(out), "rows for", out["cv_id"].nunique(), "CVs")
print("already graded:", int(out["already_graded"].sum()), "| new, need grading:", int((~out["already_graded"]).sum()))
print(out.groupby("cv_id")["already_graded"].sum().describe()[["min", "mean", "max"]].round(2).to_dict())
