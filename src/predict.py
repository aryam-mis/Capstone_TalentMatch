# src/predict.py
import argparse
import json
import os

import lightgbm as lgb
import pandas as pd


def load(path):
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    booster = lgb.Booster(model_file=os.path.join(path, "model.txt"))
    return booster, meta


def rank(booster, meta, feats, k=10):
    out = feats[["UserID", "WindowID", "JobID"]].copy()
    out["score"] = booster.predict(feats[meta["features"]])
    return out.sort_values("score", ascending=False).head(k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/reranker_v0")
    ap.add_argument("--features", required=True)
    ap.add_argument("--user", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    booster, meta = load(a.artifact)
    d = pd.read_parquet(a.features)
    uid = a.user if a.user is not None else int(d.UserID.iloc[0])
    top = rank(booster, meta, d[d.UserID == uid], a.k)
    print("artifact:", a.artifact, "| lightgbm:", meta["lightgbm"], "| user:", uid)
    print(top.to_string(index=False))