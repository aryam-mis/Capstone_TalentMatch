import os
import pandas as pd

P = "data/processed/"
OUT = "data/colab/"
os.makedirs(OUT, exist_ok=True)

# ---------- Jobs ----------
parts = []
for ch in pd.read_csv(P + "jobs_clean.tsv", sep="\t",
                      usecols=["JobID", "WindowID", "State", "text_clean"],
                      chunksize=100_000, low_memory=False, quoting=3):
    ch["text_short"] = ch["text_clean"].fillna("").astype(str).str.slice(0, 800)
    parts.append(ch.drop(columns="text_clean"))
jobs = pd.concat(parts, ignore_index=True)

# Missing states get different labels for jobs and users so they never match each other
jobs["State"] = jobs["State"].fillna("JOB_NA").astype(str).str.strip().str.upper()
assert len(jobs) == 1_092_096, len(jobs)
jobs.to_parquet(OUT + "jobs_for_emb.parquet", index=False)

# ---------- Users ----------
users = pd.read_csv(P + "users_clean.tsv", sep="\t", quoting=3)
hist = pd.read_csv(P + "user_history_clean.tsv", sep="\t", quoting=3)

past = (hist.sort_values(["UserID", "WindowID", "Sequence"])
            .groupby(["UserID", "WindowID"])["JobTitle"]
            .apply(lambda s: " ; ".join(s.dropna().astype(str).head(5))))
users = users.merge(past.rename("past_titles"), on=["UserID", "WindowID"], how="left")
users["past_titles"] = users["past_titles"].fillna("")

users["State"] = users["State"].fillna("USER_NA").astype(str).str.strip().str.upper()
for c in ["DegreeType", "Major"]:
    users[c] = users[c].fillna("").astype(str)
users["TotalYearsExperience"] = users["TotalYearsExperience"].fillna(0).clip(0, 40).astype(int)

users["profile_text"] = (
    "degree: " + users.DegreeType + ". major: " + users.Major +
    ". experience: " + users.TotalYearsExperience.astype(str) +
    " years. past roles: " + users.past_titles
).str.lower()

users = users.reset_index(drop=True)   # row i here must match row i in the embeddings
assert len(users) == 389_708, len(users)
users[["UserID", "WindowID", "State", "profile_text"]].to_parquet(
    OUT + "users_for_emb.parquet", index=False)

print("jobs:", len(jobs), "users:", len(users))