# tests/test_smoke.py
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from filters import open_jobs  # noqa: E402


def test_open_jobs_filter():
    jobs = pd.DataFrame({
        "JobID": [1, 2, 3, 4, 5],
        "WindowID": [1] * 5,
        "StartDate": ["2012-05-01", "2012-06-10", "2012-06-14", "2012-05-01", None],
        "EndDate": ["2012-06-05", "2012-06-30", "2012-06-30", None, None],
    })
    out = open_jobs(jobs, "2012-06-06", "2012-06-14")
    # 1 closed before the period, 3 starts after it; 2, 4, 5 are open
    assert sorted(out.JobID.tolist()) == [2, 4, 5]


def test_ranker_trains_and_ranks():
    rng = np.random.default_rng(0)
    n_users, n_cand = 40, 20
    x1 = rng.normal(size=n_users * n_cand)
    x2 = rng.normal(size=n_users * n_cand)
    y = (x1 + rng.normal(scale=0.5, size=x1.size) > 1.2).astype(int)
    assert y.sum() > 0

    X = pd.DataFrame({"x1": x1, "x2": x2})
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=30,
                           min_child_samples=5, random_state=0, verbose=-1)
    model.fit(X, y, group=[n_cand] * n_users)

    pred = model.predict(X)
    assert pred.shape == (n_users * n_cand,)
    assert np.corrcoef(pred, x1)[0, 1] > 0.3