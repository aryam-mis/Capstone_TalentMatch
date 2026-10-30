# src/baseline_tfidf_open.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from filters import open_jobs, VAL_START, VAL_END

K = 100
BATCH = 500
MAXCHARS = 1500
HIST = "data/processed/user_history_clean.tsv"

tr = pd.read_csv("results/split/train.csv", usecols=["UserID", "WindowID", "JobID"])
tr["UserID"] = tr["UserID"].astype(str)
tr["JobID"] = tr["JobID"].astype(str)
val_users = pd.read_csv("results/split/val.csv", usecols=["UserID"]).drop_duplicates()

users = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                    usecols=["UserID", "WindowID", "State", "DegreeType", "Major"],
                    dtype=str, keep_default_na=False)
users["UserID"] = users["UserID"].astype(str)
val_users["UserID"] = val_users["UserID"].astype(str)

vu = val_users.merge(users, on="UserID", how="left")
print("val users:", len(vu), "| without info:", vu.State.isna().sum())
vu = vu.dropna(subset=["State"])
vu = vu[vu.State != ""]

# previous job titles
hist = pd.read_csv(HIST, sep="\t", dtype=str, keep_default_na=False, on_bad_lines="skip")
title_col = [c for c in hist.columns if "title" in c.lower()][0]
hist["UserID"] = hist["UserID"].astype(str)
hist_txt = hist.groupby("UserID")[title_col].apply(lambda s: " ".join(s)).rename("htext")

vu = vu.merge(hist_txt, on="UserID", how="left").fillna({"htext": ""})
vu["utext"] = (vu.DegreeType + " " + vu.Major + " " + vu.htext).str.strip()
vu = vu[vu.utext != ""]
print("val users with text:", len(vu))

# jobs: read in chunks, keep only the windows of val users (now with dates)
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

# fit on train jobs only (no leakage), BEFORE the open filter
tr_keys = tr[["JobID", "WindowID"]].drop_duplicates()
tr_keys["WindowID"] = tr_keys["WindowID"].astype(str)
fit_jobs = jobs.merge(tr_keys, on=["JobID", "WindowID"], how="inner")
print("jobs used for fit:", len(fit_jobs))

vec = TfidfVectorizer(stop_words="english", max_features=100000,
                      ngram_range=(1, 2), min_df=3, sublinear_tf=True)
vec.fit(fit_jobs["text_clean"])

# candidates = open jobs only
jobs = open_jobs(jobs, VAL_START, VAL_END)
print("open jobs (candidates):", len(jobs))

# jobs each val user applied to in train
seen = (tr[tr.UserID.isin(set(vu.UserID))]
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
        # exclude jobs the user applied to in train
        for r, uid in enumerate(ub.UserID.values):
            cols = [pos[j] for j in seen.get(uid, ()) if j in pos]
            if cols:
                sim[r, cols] = -1
        k = min(K, sim.shape[1])
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
recs.to_csv("results/recs_tfidf_open.csv", index=False)
print("users with recs:", recs.UserID.nunique(), "| rows:", len(recs),
      "| avg per user:", round(len(recs) / recs.UserID.nunique(), 1))