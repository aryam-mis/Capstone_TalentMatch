import os
import numpy as np
import pandas as pd
import faiss

from split_utils import load_apps, temporal_split

K_FINAL = 100
K_RETRIEVE = 150          # retrieve extra so 100 remain after removing seen jobs
VAL_START = pd.Timestamp("2012-06-06")
VAL_END = pd.Timestamp("2012-06-14")
EXCLUDE_SEEN = True       # must match the baselines (check with Aryam)

os.makedirs("results/recs", exist_ok=True)

# ---------- Load embeddings and keys (row i of keys == row i of embeddings) ----------
jobs = pd.read_parquet("artifacts/jobs_keys.parquet").reset_index(drop=True)
users = pd.read_parquet("artifacts/users_keys.parquet").reset_index(drop=True)
J = np.load("artifacts/job_emb_minilm_fp16.npy", mmap_mode="r")
U = np.load("artifacts/user_emb_minilm_fp16.npy", mmap_mode="r")
assert len(J) == len(jobs) and len(U) == len(users)

# ---------- Job dates (same file and row order as the embeddings) ----------
dates = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t", quoting=3,
                    usecols=["JobID", "WindowID", "StartDate", "EndDate"],
                    low_memory=False)
assert len(dates) == len(jobs)
assert (dates["JobID"].values == jobs["JobID"].values).all()
assert (dates["WindowID"].values == jobs["WindowID"].values).all()
jobs["StartDate"] = pd.to_datetime(dates["StartDate"], errors="coerce")
jobs["EndDate"] = pd.to_datetime(dates["EndDate"], errors="coerce")

# Open during the validation period (jobs with missing dates are kept)
open_mask = ((jobs["EndDate"].isna() | (jobs["EndDate"] >= VAL_START)) &
             (jobs["StartDate"].isna() | (jobs["StartDate"] < VAL_END)))
print("jobs open during validation:", int(open_mask.sum()), "of", len(jobs))

# ---------- Validation users ----------
train, val, _ = temporal_split(load_apps())
val_pairs = val[["UserID", "WindowID"]].drop_duplicates()
print("val users:", val_pairs.UserID.nunique(), "| (user, window) pairs:", len(val_pairs))
pair_mask = pd.MultiIndex.from_frame(users[["UserID", "WindowID"]]).isin(
    pd.MultiIndex.from_frame(val_pairs))
val_users = users[pair_mask]

job_ids = jobs["JobID"].values
seen = (train[["UserID", "WindowID", "JobID"]].drop_duplicates().assign(_seen=1))


def retrieve(user_df, subset_mask, k):
    sub = jobs[subset_mask]
    cells = sub.groupby(["WindowID", "State"]).indices
    pos_in_all = np.flatnonzero(subset_mask.values)
    rows = []
    for (w, st), ug in user_df.groupby(["WindowID", "State"]):
        cidx = cells.get((w, st))
        if cidx is None:
            continue
        jidx = pos_in_all[cidx]
        index = faiss.IndexFlatIP(J.shape[1])
        index.add(np.ascontiguousarray(J[jidx], dtype="float32"))
        q = np.ascontiguousarray(U[ug.index.to_numpy()], dtype="float32")
        scores, pos = index.search(q, min(k, len(jidx)))
        for uid, s_row, p_row in zip(ug["UserID"].values, scores, pos):
            for rank, (s, p) in enumerate(zip(s_row, p_row), start=1):
                rows.append((uid, w, job_ids[jidx[p]], rank, float(s)))
    return pd.DataFrame(rows, columns=["UserID", "WindowID", "JobID", "rank", "score"])


def finalize(recs):
    if EXCLUDE_SEEN:
        recs = recs.merge(seen, on=["UserID", "WindowID", "JobID"], how="left")
        recs = recs[recs["_seen"].isna()].drop(columns="_seen")
    recs = recs.sort_values(["UserID", "WindowID", "rank"])
    recs["rank"] = recs.groupby(["UserID", "WindowID"]).cumcount() + 1
    return recs[recs["rank"] <= K_FINAL]


all_jobs = pd.Series(True, index=jobs.index)
for name, mask in [("all", all_jobs), ("open", open_mask)]:
    recs = finalize(retrieve(val_users, mask, K_RETRIEVE))
    path = f"results/recs/content_minilm_{name}.csv"
    recs[["UserID", "WindowID", "JobID", "rank"]].to_csv(path, index=False)
    print("saved", path, len(recs))