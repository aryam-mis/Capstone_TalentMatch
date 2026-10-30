# src/bootstrap_test_ci.py
# Paired user-level bootstrap on the test period: the re-ranker against the other methods
import argparse

import numpy as np
import pandas as pd

from metrics_utils import build_rel, per_user_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="docs/bootstrap_test_ci.csv")
args = ap.parse_args()

METRICS = ["P@10", "NDCG@10", "MAP@10", "Recall@100"]
FILES = {
    "reranker": "results/recs/test_rerank_v2_lam0.csv",
    "reranker_lam0.15": "results/recs/test_rerank_v2_lam0.15.csv",
    "rrf3": "results/recs/test_hybrid_rrf3.csv",
    "logreg": "results/recs/lr_v2_test.csv",
    "popularity": "results/recs/test_pop100.csv",
}

rel = build_rel(pd.read_csv("results/recs/test_labels.csv"))
pu = {}
for name, path in FILES.items():
    pu[name] = per_user_metrics(pd.read_csv(path), rel)[METRICS]
    print(name, "users", len(pu[name]), {m: round(float(v), 4) for m, v in pu[name].mean().items()})

COMPARISONS = [("reranker", "rrf3"), ("reranker", "logreg"), ("reranker", "popularity"),
               ("reranker", "reranker_lam0.15")]
rng = np.random.default_rng(args.seed)
rows = []
for a, b in COMPARISONS:
    ma_all = pu[a].values
    mb_all = pu[b].reindex(pu[a].index).values
    n = len(ma_all)
    diffs = np.empty((args.n_boot, len(METRICS)))
    gains = np.empty((args.n_boot, len(METRICS)))
    for i in range(args.n_boot):
        idx = rng.integers(0, n, n)
        ma, mb = ma_all[idx].mean(axis=0), mb_all[idx].mean(axis=0)
        diffs[i], gains[i] = ma - mb, ma / mb - 1
    for j, m in enumerate(METRICS):
        rows.append({"A": a, "B": b, "metric": m, "users": n,
                     "mean_A": ma_all[:, j].mean(), "mean_B": mb_all[:, j].mean(),
                     "diff": ma_all[:, j].mean() - mb_all[:, j].mean(),
                     "diff_ci_low": np.percentile(diffs[:, j], 2.5),
                     "diff_ci_high": np.percentile(diffs[:, j], 97.5),
                     "gain_pct": 100 * (ma_all[:, j].mean() / mb_all[:, j].mean() - 1),
                     "gain_ci_low": 100 * np.percentile(gains[:, j], 2.5),
                     "gain_ci_high": 100 * np.percentile(gains[:, j], 97.5)})

res = pd.DataFrame(rows).round(4)
res.to_csv(args.out, index=False)
pd.set_option("display.width", 250)
print(res[res["metric"] == "NDCG@10"].to_string(index=False))
print("saved", args.out)
