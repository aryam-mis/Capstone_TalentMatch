# src/baseline_tfidf_period.py
# TF-IDF baseline for any period: open jobs only, jobs applied to before the cutoff excluded
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from filters import open_jobs
from split_utils import load_apps

BATCH = 500
MAXCHARS = 1500
HIST = "data/processed/user_history_clean.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users_from", required=True)    # e.g. results/split/test.csv
    ap.add_argument("--open_from", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--history_before", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    apps = load_apps()
    ha = apps[apps.ApplicationDate < pd.Timestamp(a.history_before)][["UserID", "WindowID", "JobID"]]
    for c in ["UserID", "WindowID", "JobID"]:
        ha[c] = ha[c].astype(str)
    print("history rows:", len(ha))

    target = pd.read_csv(a.users_from, usecols=["UserID"]).drop_duplicates()
    target["UserID"] = target["UserID"].astype(str)

    users = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                        usecols=["UserID", "WindowID", "State", "DegreeType", "Major"],
                        dtype=str, keep_default_na=False)
    users["UserID"] = users["UserID"].astype(str)
    vu = target.merge(users, on="UserID", how="left")
    vu = vu.dropna(subset=["State"])
    vu = vu[vu.State != ""]

    # previous job titles of each user (profile work history)
    ht = pd.read_csv(HIST, sep="\t", dtype=str, keep_default_na=False, on_bad_lines="skip")
    title_col = [c for c in ht.columns if "title" in c.lower()][0]
    ht["UserID"] = ht["UserID"].astype(str)
    ht_txt = ht.groupby("UserID")[title_col].apply(lambda s: " ".join(s)).rename("htext")

    vu = vu.merge(ht_txt, on="UserID", how="left").fillna({"htext": ""})
    vu["utext"] = (vu.DegreeType + " " + vu.Major + " " + vu.htext).str.strip()
    vu = vu[vu.utext != ""]
    print("target users with text:", len(vu))

    # jobs of the users' windows (text truncated to keep memory low)
    windows = set(vu.WindowID.unique())
    parts = []
    for ch in pd.read_csv("data/processed/jobs_clean.tsv", sep="\t", dtype=str,
                          keep_default_na=False, chunksize=100000,
                          usecols=["JobID", "WindowID", "State", "text_clean",
                                   "StartDate", "EndDate"]):
        ch = ch[ch.WindowID.isin(windows)]
        ch["text_clean"] = ch["text_clean"].str.slice(0, MAXCHARS)
        parts.append(ch)
    jobs = pd.concat(parts, ignore_index=True)
    print("jobs loaded:", len(jobs))

    # fit on jobs applied to before the cutoff only (no leakage), BEFORE the open filter
    keys = ha[["JobID", "WindowID"]].drop_duplicates()
    fit_jobs = jobs.merge(keys, on=["JobID", "WindowID"], how="inner")
    print("jobs used for fit:", len(fit_jobs))

    vec = TfidfVectorizer(stop_words="english", max_features=100000,
                          ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    vec.fit(fit_jobs["text_clean"])

    jobs = open_jobs(jobs, a.open_from, a.end)
    print("open jobs (candidates):", len(jobs))

    seen = (ha[ha.UserID.isin(set(vu.UserID))]
            .groupby("UserID")["JobID"].apply(set).to_dict())

    out = []
    for (w, s), ug in vu.groupby(["WindowID", "State"]):
        jg = jobs[(jobs.WindowID == w) & (jobs.State == s)]
        if jg.empty:
            continue
        J = vec.transform(jg["text_clean"])
        job_ids = jg["JobID"].values
        pos = {j: i for i, j in enumerate(job_ids)}
        for i in range(0, len(ug), BATCH):
            ub = ug.iloc[i:i + BATCH]
            U = vec.transform(ub["utext"])
            sim = (U @ J.T).toarray()
            for r, uid in enumerate(ub.UserID.values):
                cols = [pos[j] for j in seen.get(uid, ()) if j in pos]
                if cols:
                    sim[r, cols] = -1
            k = min(a.k, sim.shape[1])
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            for r in range(sim.shape[0]):
                idx = top[r][np.argsort(-sim[r, top[r]])]
                idx = idx[sim[r, idx] >= 0]
                if len(idx) == 0:
                    continue
                out.append(pd.DataFrame({
                    "UserID": ub.UserID.iloc[r], "WindowID": w,
                    "JobID": job_ids[idx], "rank": np.arange(1, len(idx) + 1)}))

    recs = pd.concat(out, ignore_index=True)
    recs.to_csv(a.out, index=False)
    print("users with recs:", recs.UserID.nunique(), "| rows:", len(recs),
          "| avg per user:", round(len(recs) / recs.UserID.nunique(), 1))


if __name__ == "__main__":
    main()