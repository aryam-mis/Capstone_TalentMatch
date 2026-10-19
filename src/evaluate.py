import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["UserID", "WindowID", "JobID"]
KS_RECALL = (50, 100)


def load_truth(split):
    d = Path("results/split")
    truth = pd.read_csv(d / f"{split}.csv", usecols=KEY).drop_duplicates()
    seen_files = {"val": ["train"], "test": ["train", "val"]}[split]
    seen = pd.concat([pd.read_csv(d / f"{s}.csv", usecols=KEY) for s in seen_files]).drop_duplicates()
    truth = truth.merge(seen, on=KEY, how="left", indicator=True)
    truth = truth[truth["_merge"] == "left_only"].drop(columns="_merge")
    return truth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recs", required=True, help="CSV: UserID, WindowID, JobID, rank (rank optional)")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--name", default="", help="method name, appended to results/metrics.csv")
    args = p.parse_args()

    truth = load_truth(args.split)
    rel = {}
    for u, w, j in zip(truth.UserID, truth.WindowID, truth.JobID):
        rel.setdefault(u, set()).add((j, w))

    recs = pd.read_csv(args.recs)
    if "rank" in recs.columns:
        recs = recs.sort_values(["UserID", "rank"])
    recs = recs.drop_duplicates(["UserID", "WindowID", "JobID"])
    rec_lists = {u: list(zip(g.JobID, g.WindowID))
                 for u, g in recs.groupby("UserID", sort=False)}

    rows = []
    for u, rset in rel.items():
        lst = rec_lists.get(u, [])
        hits = np.array([1 if x in rset else 0 for x in lst[:100]])
        h10 = hits[:10]
        n_rel = len(rset)
        m = min(n_rel, 10)
        disc = 1.0 / np.log2(np.arange(2, 12))
        dcg = float((h10 * disc[:len(h10)]).sum())
        idcg = float(disc[:m].sum())
        prec_at = np.cumsum(h10) / np.arange(1, len(h10) + 1) if len(h10) else np.array([])
        ap = float((prec_at * h10).sum() / m) if len(h10) else 0.0
        row = {"P@10": h10.sum() / 10, "NDCG@10": dcg / idcg, "MAP@10": ap}
        for k in KS_RECALL:
            row[f"Recall@{k}"] = hits[:k].sum() / n_rel
        rows.append(row)

    res = pd.DataFrame(rows).mean()
    n_users = len(rel)
    n_with_recs = sum(1 for u in rel if u in rec_lists)
    print(f"split={args.split} | users evaluated: {n_users} | users with recs: {n_with_recs}")
    print(res.round(4).to_string())

    if args.name:
        out = Path("results/metrics.csv")
        line = pd.DataFrame([{"method": args.name, "split": args.split, "users": n_users, **res.round(4).to_dict()}])
        line.to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
