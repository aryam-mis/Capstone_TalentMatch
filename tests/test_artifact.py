# tests/test_artifact.py
import hashlib
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "reranker_v2")


def _meta():
    with open(os.path.join(ART, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def test_checksum_matches_metadata():
    meta = _meta()
    with open(os.path.join(ART, meta["model_file"]), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == meta["model_sha256"]


def test_loads_and_ranks_cold_start_rows():
    meta = _meta()
    booster = lgb.Booster(model_file=os.path.join(ART, meta["model_file"]))
    assert booster.num_feature() == len(meta["features"])
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(50, len(meta["features"]))),
                     columns=meta["features"]).astype("float32")
    x.loc[:, ["collab_score", "collab_rank"]] = np.nan      # new user: no collaborative signal
    pred = booster.predict(x)
    assert pred.shape == (50,) and np.isfinite(pred).all()