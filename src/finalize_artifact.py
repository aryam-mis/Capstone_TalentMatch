# src/finalize_artifact.py
# Makes the model artifact self-describing: version, commit, checksum, configuration and results
import argparse
import hashlib
import json
import os
import subprocess
from datetime import date

import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/reranker_v2")
    a = ap.parse_args()

    meta_path = os.path.join(a.artifact, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    m = pd.read_csv("results/metrics.csv")
    m = m.drop_duplicates(subset=["method", "split"], keep="last").set_index("method")

    def row(name):
        r = m.loc[name]
        return {k: float(r[k]) for k in ["P@10", "NDCG@10", "MAP@10", "Recall@50", "Recall@100"]}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None

    meta.update({
        "name": "talentmatch-reranker",
        "version": "2.0",
        "created": str(date.today()),
        "git_commit_at_creation": commit,
        "model_file": "model.txt",
        "model_sha256": sha256(os.path.join(a.artifact, "model.txt")),
        "task": "LightGBM LambdaRank re-ranker over content, collaborative and popularity candidates",
        "candidates": {"content_k": 100, "collab_k": 100, "popularity_k": 50},
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "periods": {"train": "2012-05-30..2012-06-05", "val": "2012-06-06..2012-06-13",
                    "test": "2012-06-14..2012-06-26 (open jobs from 2012-06-18)"},
        "diversity_lambda_frozen": 0.15,
        "results": {"val_lambda0": row("rerank_v2"),
                    "test_lambda0": row("test_rerank_v2_lam0"),
                    "test_lambda0.15": row("test_rerank_v2_lam0.15")},
        "usage": "python src/profile_recommender.py --state CA --degree \"Bachelor's\" "
                 "--major \"Business Administration\" --years 5 --roles \"Sales Associate\"",
    })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("updated", meta_path)
    print("model sha256:", meta["model_sha256"])


if __name__ == "__main__":
    main()