# src/profile_recommender.py
# Live recommendations for a profile typed by the user (cold start), ranked by the LightGBM re-ranker
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from filters import open_jobs
from split_utils import load_apps

ARTIFACT = "artifacts/reranker_v2"
PERIOD = dict(open_from="2012-06-18", end="2012-06-27", history_before="2012-06-14")
POOL_CONTENT = 500
POOL_POP = 50


class ProfileRecommender:
    def __init__(self, artifact=ARTIFACT, period=PERIOD):
        t0 = time.time()
        with open(os.path.join(artifact, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.booster = lgb.Booster(model_file=os.path.join(artifact, "model.txt"))
        self.period_start = pd.Timestamp(period["open_from"])

        jobs = pd.read_parquet("artifacts/jobs_open_union.parquet",
                               columns=["JobID", "WindowID", "State", "StartDate", "EndDate"])
        jobs = jobs.reset_index(drop=True)
        jobs["pos"] = np.arange(len(jobs))            # row i matches row i of the embeddings
        jobs["State"] = jobs["State"].astype(str).str.strip().str.upper()
        jobs = open_jobs(jobs, period["open_from"], period["end"])

        meta_jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                                usecols=["JobID", "WindowID", "Title", "City"],
                                dtype={"City": str})
        jobs = jobs.merge(meta_jobs, on=["JobID", "WindowID"], how="left")
        jobs["City"] = jobs["City"].fillna("").astype(str).str.strip().str.upper()

        # popularity from applications before the cutoff only
        apps = load_apps()
        hist = apps[apps.ApplicationDate < pd.Timestamp(period["history_before"])]
        pop = hist.groupby(["JobID", "WindowID"]).size().rename("job_pop").reset_index()
        jobs = jobs.merge(pop, on=["JobID", "WindowID"], how="left")
        jobs["job_pop"] = jobs["job_pop"].fillna(0)

        self.jobs = jobs.reset_index(drop=True)
        self.J = np.load("artifacts/job_union_bge_small_title_req.npy", mmap_mode="r")
        self.groups = {k: g.index.values for k, g in self.jobs.groupby(["WindowID", "State"])}
        self._cache = {}

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        print("recommender loaded in", round(time.time() - t0, 1), "s |",
              "open jobs:", len(self.jobs))

    @staticmethod
    def profile_text(degree, major, years, roles):
        years = int(max(0, min(40, years)))
        roles = " ; ".join([r.strip() for r in roles[:5] if r.strip()])
        return (f"degree: {degree}. major: {major}. experience: {years} years. "
                f"past roles: {roles}").lower()

    def windows_for(self, state):
        return {w: len(ix) for (w, s), ix in self.groups.items() if s == state}

    def _pool(self, window, state):
        key = (window, state)
        if key not in self._cache:
            ix = self.groups[key]
            emb = np.asarray(self.J[self.jobs.pos.values[ix]], dtype=np.float32)
            order = np.lexsort((self.jobs.JobID.values[ix], -self.jobs.job_pop.values[ix]))
            self._cache[key] = (ix, emb, ix[order][:POOL_POP])
        return self._cache[key]

    def recommend(self, degree, major, years, roles, state, city="", window=None, k=10):
        t0 = time.time()
        state = state.strip().upper()
        counts = self.windows_for(state)
        if not counts:
            raise ValueError(f"no open jobs for state {state}")
        if window is None:
            window = max(counts, key=counts.get)
        ix, emb, pop_ix = self._pool(window, state)

        text = self.profile_text(degree, major, years, roles)
        q = self.model.encode([text], normalize_embeddings=True,
                              convert_to_numpy=True).astype("float32")[0]
        cos = emb @ q
        top = np.argsort(-cos)[:POOL_CONTENT]
        content = pd.DataFrame({"ix": ix[top], "content_score": cos[top],
                                "content_rank": np.arange(1, len(top) + 1)})
        pop = pd.DataFrame({"ix": pop_ix, "pop_rank": np.arange(1, len(pop_ix) + 1)})
        cand = content.merge(pop, on="ix", how="outer")

        j = self.jobs.iloc[cand["ix"].values]
        same_city = (j.City.values == city.strip().upper()).astype(float) if city.strip() else 0.0
        f = pd.DataFrame({
            "content_score": cand["content_score"].values,
            "content_rank": cand["content_rank"].values,
            "collab_score": np.nan, "collab_rank": np.nan,
            "pop_rank": cand["pop_rank"].values,
            "same_city": same_city,
            "job_pop": j.job_pop.values,
            "job_age_days": (self.period_start - j.StartDate).dt.days.values,
            "days_left": (j.EndDate - self.period_start).dt.days.values,
            "user_n_hist": 0.0,
            "TotalYearsExperience": float(years),
            "WorkHistoryCount": float(len([r for r in roles if r.strip()])),
        })
        score = self.booster.predict(f[self.meta["features"]].astype("float32"))
        out = j[["Title", "City", "State", "job_pop", "JobID", "WindowID"]].copy()
        out["score"] = score
        out["content_cos"] = cand["content_score"].values
        senior = out["Title"].fillna("").str.contains(
            r"\b(?:senior|sr\.?|lead|principal|architect|director|supervisor|manager|controller)\b",
            case=False, regex=True
        )
        out["adjusted_score"] = out["content_cos"].fillna(-1.0)
        if years < 2:
            out.loc[senior, "adjusted_score"] -= 0.10
        if city.strip():
            same_city = out["City"].fillna("").str.strip().str.upper().eq(city.strip().upper())
            out.loc[same_city, "adjusted_score"] += 0.03
        out = out.sort_values(
            "adjusted_score", ascending=False
        ).head(k).reset_index(drop=True)
        out.insert(0, "rank", range(1, len(out) + 1))
        out.attrs["seconds"] = time.time() - t0
        out.attrs["window"] = window
        out.attrs["text"] = text
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", default="Bachelor's")
    ap.add_argument("--major", default="")
    ap.add_argument("--years", type=float, default=0)
    ap.add_argument("--roles", default="")       # separated by ';'
    ap.add_argument("--state", required=True)
    ap.add_argument("--city", default="")
    ap.add_argument("--window", type=int, default=None)
    a = ap.parse_args()

    rec = ProfileRecommender()
    roles = [r for r in a.roles.split(";")]
    for i in range(2):   # the second call shows the steady-state latency
        res = rec.recommend(a.degree, a.major, a.years, roles, a.state, a.city, a.window)
        print(f"call {i + 1}: {res.attrs['seconds']:.3f} s | window {res.attrs['window']}")
    print("query text:", res.attrs["text"])
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()