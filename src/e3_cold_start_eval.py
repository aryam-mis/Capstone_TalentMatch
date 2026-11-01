# src/e3_cold_start_eval.py
# Cold start on the TEST period, evaluated separately for new users and new jobs (Track C brief).
#  (1) New users: metrics by number of prior applications (0 / 1-2 / 3+) before 2012-06-14.
#  (2) New jobs: recall of test targets whose job had no application before 2012-06-14.
# "profile_match" is the app's cold-start mode applied offline: the content retriever list
# (same bge-small profile/job embeddings) re-sorted with the senior-title penalty for users
# with fewer than 3 years. The typed skills text is not available for dataset users.
# Uses existing files only: no model is retrained and the frozen test runs are only read.
import argparse

import numpy as np
import pandas as pd

from metrics_utils import build_rel, per_user_metrics
from profile_recommender import JUNIOR_YEARS, SENIOR_PENALTY, SENIOR_RE
from split_utils import load_apps

KEY = ["UserID", "WindowID", "JobID"]
TEST_START = pd.Timestamp("2012-06-14")
BINS = [("0 (new user)", 0, 0), ("1-2", 1, 2), ("3+", 3, 10**9)]
METRICS = ["P@10", "NDCG@10", "MAP@10", "Recall@100"]
FILES = {
    "popularity": "results/recs/test_pop100.csv",
    "tfidf": "results/recs/test_tfidf_open.csv",
    "content": "results/recs/test_content.csv",
    "collab": "results/recs/test_collab.csv",
    "hybrid_rrf3": "results/recs/test_hybrid_rrf3.csv",
    "logreg": "results/recs/lr_v2_test.csv",
    "reranker": "results/recs/test_rerank_v2_lam0.csv",
}

ap = argparse.ArgumentParser()
ap.add_argument("--out_users", default="docs/e3_cold_users_test.csv")
ap.add_argument("--out_jobs", default="docs/e3_cold_jobs_test.csv")
args = ap.parse_args()

# ---------- ground truth and history (only applications before the test period) ----------
labels = pd.read_csv("results/recs/test_labels.csv")
rel = build_rel(labels)
apps = load_apps()
hist = apps[apps["ApplicationDate"] < TEST_START][KEY].drop_duplicates()
n_hist = hist.groupby(["UserID", "WindowID"]).size().rename("n_hist").reset_index()
pairs = (labels[["UserID", "WindowID"]].drop_duplicates()
         .merge(n_hist, on=["UserID", "WindowID"], how="left").fillna({"n_hist": 0}))
n_of = dict(zip(pairs["UserID"], pairs["n_hist"]))
print("history rows before test:", len(hist), "(expected 1,444,385) | test users:", len(rel))

seen_jobs = set(zip(hist["JobID"], hist["WindowID"]))
labels["new_job"] = [(j, w) not in seen_jobs for j, w in zip(labels["JobID"], labels["WindowID"])]
rel_new_jobs = build_rel(labels[labels["new_job"]])
print("test targets on new jobs (no application before 06-14):",
      f"{labels['new_job'].mean():.1%} | users with such a target: {len(rel_new_jobs)}")

# ---------- profile_match: content list + senior-title penalty for juniors ----------
content = pd.read_csv(FILES["content"])
titles = pd.read_parquet("data/processed/jobs_text_open.parquet", columns=["JobID", "WindowID", "Title"])
titles = titles.drop_duplicates(["JobID", "WindowID"])
years = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                    usecols=["UserID", "WindowID", "TotalYearsExperience"]).drop_duplicates(["UserID", "WindowID"])
pm = content.merge(titles, on=["JobID", "WindowID"], how="left").merge(years, on=["UserID", "WindowID"], how="left")
junior = pm["TotalYearsExperience"].fillna(0) < JUNIOR_YEARS
senior = pm["Title"].fillna("").str.contains(SENIOR_RE, regex=True)
pm["score"] = pm["score"] - SENIOR_PENALTY * (junior & senior)
pm = pm.sort_values(["UserID", "score"], ascending=[True, False], kind="stable")
pm["rank"] = pm.groupby("UserID").cumcount() + 1
recs_all = {"profile_match": pm[KEY + ["rank", "score"]]}
recs_all.update({name: (content if name == "content" else pd.read_csv(path))
                 for name, path in FILES.items()})

# ---------- (1) new users: metrics by history length ----------
rows = []
for name, recs in recs_all.items():
    pu = per_user_metrics(recs, rel)[METRICS]
    pu["n_hist"] = pu.index.map(n_of)
    for label, lo, hi in BINS:
        g = pu[(pu["n_hist"] >= lo) & (pu["n_hist"] <= hi)]
        rows.append({"method": name, "history": label, "users": len(g),
                     **g[METRICS].mean().round(4).to_dict()})
    rows.append({"method": name, "history": "all", "users": len(pu),
                 **pu[METRICS].mean().round(4).to_dict()})
users_tab = pd.DataFrame(rows)
users_tab.to_csv(args.out_users, index=False)

# ---------- (2) new jobs: recall of targets on jobs without any earlier application ----------
rows = []
for name, recs in recs_all.items():
    r = recs.sort_values(["UserID", "rank"])
    lists = {u: list(zip(g["JobID"], g["WindowID"])) for u, g in r.groupby("UserID", sort=False)}
    res = {k: [] for k in (10, 50, 100)}
    for u, rset in rel_new_jobs.items():
        lst = lists.get(u, [])
        for k in res:
            res[k].append(len(set(lst[:k]) & rset) / len(rset))
    rows.append({"method": name, "users": len(rel_new_jobs),
                 **{f"NewJobRecall@{k}": round(float(np.mean(v)), 4) for k, v in res.items()}})
jobs_tab = pd.DataFrame(rows)
jobs_tab.to_csv(args.out_jobs, index=False)

pd.set_option("display.width", 200)
print("\nNEW USERS (test, by prior applications):")
print(users_tab.pivot(index="method", columns="history", values="NDCG@10")
      .reindex(columns=[b[0] for b in BINS] + ["all"]).round(4).to_string())
print("\nusers per group:", users_tab[users_tab.method == "reranker"][["history", "users"]]
      .to_dict("records"))
print("\nNEW JOBS (test):")
print(jobs_tab.to_string(index=False))
print("\nsaved", args.out_users, "and", args.out_jobs)
