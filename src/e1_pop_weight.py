import argparse

import pandas as pd

from metrics_utils import load_val_context, rrf, per_user_metrics, cold_recall, share_not_in_content

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

p = argparse.ArgumentParser()
p.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 1.0])
args = p.parse_args()

rel, group_of, rel_cold = load_val_context()
content = pd.read_csv("results/recs/val_content.csv")
collab = pd.read_csv("results/recs/val_collab.csv")
pop = pd.read_csv("results/recs/val_pop_state_open.csv")

variants = {"content+collab+pop": 1.0, "content+pop": 0.0}   # variant -> collab weight
rows = []
for variant, wc in variants.items():
    for wp in args.weights:
        hyb = rrf([(content, 1.0), (collab, wc), (pop, wp)])
        pu = per_user_metrics(hyb, rel)
        g = pu.index.map(group_of)
        r100 = pu["Recall@100"]
        cj = cold_recall(hyb, rel_cold, 100)
        row = {"variant": variant, "pop_weight": wp,
               "P@10": pu["P@10"].mean(), "NDCG@10": pu["NDCG@10"].mean(),
               "R@100 all users": r100.mean(),
               "R@100 cold-start users": r100[g == 0].mean(),
               "R@100 1-2": r100[(g >= 1) & (g <= 2)].mean(),
               "R@100 3+": r100[g >= 3].mean(),
               "R@100 new jobs": cj.mean(),
               "top10 share not from content": share_not_in_content(hyb, content, 10)}
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        print(rows[-1], flush=True)

res = pd.DataFrame(rows)
res.to_csv("results/e1_pop_weight.csv", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))

if plt is not None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (variant, d) in zip(axes, res.groupby("variant")):
        ax.plot(d["pop_weight"], d["R@100 all users"], marker="o", label="all users")
        ax.plot(d["pop_weight"], d["R@100 cold-start users"], marker="o", label="users without history")
        ax.plot(d["pop_weight"], d["R@100 new jobs"], marker="o", label="new jobs")
        ax.set_title(variant)
        ax.set_xlabel("popularity weight in RRF")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall@100")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig("results/e1_pop_weight.png", dpi=150)
    print("saved results/e1_pop_weight.png")