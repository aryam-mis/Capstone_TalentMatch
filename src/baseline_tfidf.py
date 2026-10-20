# src/baseline_tfidf.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

K = 150
BATCH = 500
MAXCHARS = 1500
HIST = "data/processed/user_history_clean.tsv"

tr = pd.read_csv("results/split/train.csv", usecols=["UserID", "WindowID", "JobID"])
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

# ط¹ظ†ط§ظˆظٹظ† ط§ظ„ظˆط¸ط§ط¦ظپ ط§ظ„ط³ط§ط¨ظ‚ط©
hist = pd.read_csv(HIST, sep="\t", dtype=str, keep_default_na=False, on_bad_lines="skip")
print("user_history columns:", hist.columns.tolist())
title_col = [c for c in hist.columns if "title" in c.lower()][0]
hist["UserID"] = hist["UserID"].astype(str)
hist_txt = hist.groupby("UserID")[title_col].apply(lambda s: " ".join(s)).rename("htext")

vu = vu.merge(hist_txt, on="UserID", how="left").fillna({"htext": ""})
vu["utext"] = (vu.DegreeType + " " + vu.Major + " " + vu.htext).str.strip()
vu = vu[vu.utext != ""]
print("val users with text:", len(vu))

# ط§ظ„ظˆط¸ط§ط¦ظپ: ظ†ظ‚ط±ط£ ط¨ط¯ظپط¹ط§طھ ظˆظ†ظ‚ط·ط¹ ط§ظ„ظ†طµطŒ ظˆظ†ط­طھظپط¸ ط¨ظ†ظˆط§ظپط° ظ…ط³طھط®ط¯ظ…ظٹ val ظپظ‚ط·
windows = set(vu.WindowID.unique())
parts = []
for ch in pd.read_csv("data/processed/jobs_clean.tsv", sep="\t", dtype=str,
                      keep_default_na=False, chunksize=100000,
                      usecols=["JobID", "WindowID", "State", "text_clean"]):
    ch = ch[ch.WindowID.isin(windows)]
    ch["text_clean"] = ch["text_clean"].str.slice(0, MAXCHARS)
    parts.append(ch)
jobs = pd.concat(parts, ignore_index=True)
print("jobs loaded:", len(jobs))

# fit ط¹ظ„ظ‰ ظˆط¸ط§ط¦ظپ train ظپظ‚ط· (ظ„ط§ leak)
tr_keys = tr[["JobID", "WindowID"]].drop_duplicates()
tr_keys["JobID"] = tr_keys["JobID"].astype(str)
tr_keys["WindowID"] = tr_keys["WindowID"].astype(str)
fit_jobs = jobs.merge(tr_keys, on=["JobID", "WindowID"], how="inner")
print("jobs used for fit:", len(fit_jobs))

vec = TfidfVectorizer(stop_words="english", max_features=100000,
                      ngram_range=(1, 2), min_df=3, sublinear_tf=True)
vec.fit(fit_jobs["text_clean"])

out = []
for (w, s), ug in vu.groupby(["WindowID", "State"]):
    jg = jobs[(jobs.WindowID == w) & (jobs.State == s)]
    if jg.empty:
        continue
    J = vec.transform(jg["text_clean"])
    job_ids = jg["JobID"].values
    for i in range(0, len(ug), BATCH):
        ub = ug.iloc[i:i + BATCH]
        U = vec.transform(ub["utext"])
        sim = (U @ J.T).toarray()
        k = min(K, sim.shape[1])
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        for r in range(sim.shape[0]):
            idx = top[r][np.argsort(-sim[r, top[r]])]
            out.append(pd.DataFrame({
                "UserID": ub.UserID.iloc[r], "WindowID": w,
                "JobID": job_ids[idx], "rank": np.arange(1, len(idx) + 1)}))

recs = pd.concat(out, ignore_index=True)
recs.to_csv("results/recs_tfidf.csv", index=False)
print("users with recs:", recs.UserID.nunique(), "| rows:", len(recs))
