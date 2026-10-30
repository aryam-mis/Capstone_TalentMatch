import argparse

import numpy as np
import pandas as pd

from metrics_utils import load_val_context, per_user_metrics, cold_recall

p = argparse.ArgumentParser()
p.add_argument("--n_boot", type=int, default=2000)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

rel, group_of, rel_cold = load_val_context()
files = {"pop": "results/recs/val_pop_state_open.csv",
         "hybrid3": "results/recs/val_hybrid_rrf3.csv",
         "content": "results/recs/val_content.csv"}
recs = {k: pd.read_csv(v) for k, v in files.items()}
pu = {k: per_user_metrics(r, rel) for k, r in recs.items()}
cj = {k: cold_recall(r, rel_cold, 100) for k, r in recs.items()}
print("sanity (Recall@100, all users):", {k: round(v["Recall@100"].mean(), 4) for k, v in pu.items()})


def paired_bootstrap(a, b, n_boot, rng):
    a, b = a.align(b, join="inner")
    av, bv = a.values, b.values
    n = len(av)
    ma, mb = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ma[i], mb[i] = av[idx].mean(), bv[idx].mean()
    d = ma - mb
    return {"users": n, "A": av.mean(), "B": bv.mean(), "diff_A_minus_B": av.mean() - bv.mean(),
            "diff_ci_low": np.percentile(d, 2.5), "diff_ci_high": np.percentile(d, 97.5),
            "A_ci": f"[{np.percentile(ma, 2.5):.4f}, {np.percentile(ma, 97.5):.4f}]",
            "B_ci": f"[{np.percentile(mb, 2.5):.4f}, {np.percentile(mb, 97.5):.4f}]"}


cold_users = [u for u in pu["pop"].index if group_of.get(u, 0) == 0]
comparisons = [
    ("A: hybrid(content+collab+pop) vs pop, Recall@100, all users",
     pu["hybrid3"]["Recall@100"], pu["pop"]["Recall@100"]),
    ("B: hybrid(content+collab+pop) vs pop, Recall@100, users without history",
     pu["hybrid3"]["Recall@100"].loc[cold_users], pu["pop"]["Recall@100"].loc[cold_users]),
    ("C: content vs hybrid(content+collab+pop), Recall@100, new-job targets",
     cj["content"], cj["hybrid3"]),
]

rng = np.random.default_rng(args.seed)
rows = []
for label, a, b in comparisons:
    r = {"comparison": label, **paired_bootstrap(a, b, args.n_boot, rng)}
    rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

res = pd.DataFrame(rows)
res.to_csv("results/bootstrap_ci.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print(res.to_string(index=False))