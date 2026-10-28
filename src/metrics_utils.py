import numpy as np
import pandas as pd

KEY = ["UserID", "WindowID", "JobID"]
DISC = 1.0 / np.log2(np.arange(2, 12))


def build_rel(labels):
    rel = {}
    for u, w, j in zip(labels["UserID"], labels["WindowID"], labels["JobID"]):
        rel.setdefault(u, set()).add((j, w))
    return rel


def load_val_context(labels_path="results/recs/val_labels.csv"):
    """Validation ground truth, history length per user, and the new-job targets."""
    from split_utils import load_apps, temporal_split

    train, _, _ = temporal_split(load_apps())
    labels = pd.read_csv(labels_path)
    rel = build_rel(labels)

    n_hist = (train[KEY].drop_duplicates().groupby(["UserID", "WindowID"])
              .size().rename("n_hist").reset_index())
    pairs = (labels[["UserID", "WindowID"]].drop_duplicates()
             .merge(n_hist, on=["UserID", "WindowID"], how="left")
             .fillna({"n_hist": 0}))
    group_of = dict(zip(pairs["UserID"], pairs["n_hist"]))

    train_jobs = set(zip(train["JobID"], train["WindowID"]))
    is_new = np.array([(j, w) not in train_jobs
                       for j, w in zip(labels["JobID"], labels["WindowID"])])
    rel_cold = build_rel(labels[is_new])
    return rel, group_of, rel_cold


def rrf(frames_weights, k_final=100, rrf_k=60):
    """Weighted reciprocal rank fusion. frames_weights = [(dataframe, weight), ...]."""
    parts = []
    for f, w in frames_weights:
        if w == 0:
            continue
        t = f[KEY + ["rank"]].copy()
        t["s"] = w / (rrf_k + t["rank"])
        parts.append(t[KEY + ["s"]])
    m = pd.concat(parts).groupby(KEY, as_index=False)["s"].sum()
    m = m.sort_values(["UserID", "WindowID", "s"], ascending=[True, True, False], kind="stable")
    m["rank"] = m.groupby(["UserID", "WindowID"]).cumcount() + 1
    m = m[m["rank"] <= k_final].rename(columns={"s": "score"})
    return m[KEY + ["rank", "score"]]


def _lists(recs):
    recs = recs.sort_values(["UserID", "rank"]).drop_duplicates(["UserID", "WindowID", "JobID"])
    return {u: list(zip(g["JobID"], g["WindowID"])) for u, g in recs.groupby("UserID", sort=False)}


def per_user_metrics(recs, rel):
    """Same definitions as evaluate.py, one row per validation user."""
    lists = _lists(recs)
    out = {}
    for u, rset in rel.items():
        lst = lists.get(u, [])
        hits = np.array([1 if x in rset else 0 for x in lst[:100]])
        h10 = hits[:10]
        n_rel = len(rset)
        m = min(n_rel, 10)
        dcg = float((h10 * DISC[:len(h10)]).sum())
        idcg = float(DISC[:m].sum())
        prec = np.cumsum(h10) / np.arange(1, len(h10) + 1) if len(h10) else np.array([])
        ap = float((prec * h10).sum() / m) if len(h10) else 0.0
        out[u] = {"P@10": h10.sum() / 10, "NDCG@10": dcg / idcg, "MAP@10": ap,
                  "Recall@50": hits[:50].sum() / n_rel, "Recall@100": hits[:100].sum() / n_rel}
    return pd.DataFrame.from_dict(out, orient="index")


def cold_recall(recs, rel_cold, k=100):
    """Recall@k per user, counting only targets whose job has no train application."""
    lists = _lists(recs)
    return pd.Series({u: len(set(lists.get(u, [])[:k]) & rset) / len(rset)
                      for u, rset in rel_cold.items()})


def share_not_in_content(recs, content, k=10):
    """Share of the top-k rows that the content retriever did not return at all."""
    top = recs[recs["rank"] <= k][KEY]
    m = top.merge(content[KEY].assign(_c=1), on=KEY, how="left")
    return float(m["_c"].isna().mean())