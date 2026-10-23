import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import faiss
import scipy.sparse as sp

from split_utils import load_apps

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="period name, e.g. train_rr, val, test")
p.add_argument("--start", required=True, help="target period start (inclusive)")
p.add_argument("--end", required=True, help="target period end (exclusive)")
p.add_argument("--history_before", required=True,
               help="collaborative matrix and seen-jobs exclusion use applications before this date")
p.add_argument("--open_from", default=None, help="jobs must end on/after this date (default: start)")
p.add_argument("--k", type=int, default=100)
p.add_argument("--eval_split", choices=["val", "test"], default=None)
p.add_argument("--jobs_parquet", default="artifacts/jobs_open_union.parquet")
p.add_argument("--jobs_emb", default="artifacts/job_union_bge_small_title_req.npy")
p.add_argument("--users_emb", default="artifacts/user_emb_bge_small.npy")
args = p.parse_args()

START, END = pd.Timestamp(args.start), pd.Timestamp(args.end)
HIST_END = pd.Timestamp(args.history_before)
OPEN_FROM = pd.Timestamp(args.open_from) if args.open_from else START
K = args.k
K_RETRIEVE = int(K * 1.5)
KEY = ["UserID", "WindowID", "JobID"]
CHUNK = 500
os.makedirs("results/recs", exist_ok=True)

# ---------- Applications: history (before history_before) and target period ----------
apps = load_apps()
d = apps["ApplicationDate"]
hist = apps[d < HIST_END][KEY].drop_duplicates()
target = apps[(d >= START) & (d < END)][KEY].drop_duplicates()
seen = hist.assign(_seen=1)

labels = target.merge(seen, on=KEY, how="left")
labels = labels[labels["_seen"].isna()].drop(columns="_seen")
labels.to_csv(f"results/recs/{args.name}_labels.csv", index=False)
print(f"[{args.name}] history rows: {len(hist)} | target rows: {len(target)} | "
      f"label pairs: {len(labels)} | label users: {labels['UserID'].nunique()}")

# ---------- Jobs open in this period ----------
jobs = pd.read_parquet(args.jobs_parquet).reset_index(drop=True)
J = np.load(args.jobs_emb, mmap_mode="r")
assert len(J) == len(jobs), (len(J), len(jobs))
is_open = ((jobs["EndDate"].isna() | (jobs["EndDate"] >= OPEN_FROM)) &
           (jobs["StartDate"].isna() | (jobs["StartDate"] < END)))
jobs["is_open"] = is_open
print("open jobs for this period:", int(is_open.sum()))
job_ids = jobs["JobID"].values

# ---------- Target users (row position in users_keys == row in the user embeddings) ----------
users = pd.read_parquet("artifacts/users_keys.parquet").reset_index(drop=True)
U = np.load(args.users_emb, mmap_mode="r")
assert len(U) == len(users), (len(U), len(users))
pairs = target[["UserID", "WindowID"]].drop_duplicates()
mask = pd.MultiIndex.from_frame(users[["UserID", "WindowID"]]).isin(
    pd.MultiIndex.from_frame(pairs))
target_users = users[mask]
print("target (user, window) pairs with a profile row:", len(target_users))


def finalize(recs):
    recs = recs.merge(seen, on=KEY, how="left")
    recs = recs[recs["_seen"].isna()].drop(columns="_seen")
    recs = recs.sort_values(["UserID", "WindowID", "rank"])
    recs["rank"] = recs.groupby(["UserID", "WindowID"]).cumcount() + 1
    return recs[recs["rank"] <= K][KEY + ["rank", "score"]]


def save_and_eval(recs, suffix):
    out = f"results/recs/{args.name}_{suffix}.csv"
    recs.to_csv(out, index=False)
    print("saved", out, len(recs), "| users with recs:",
          recs[["UserID", "WindowID"]].drop_duplicates().shape[0])
    if args.eval_split:
        subprocess.run([sys.executable, "src/evaluate.py", "--recs", out,
                        "--split", args.eval_split, "--name", f"{args.name}_{suffix}"],
                       check=True)


# ====================== Content retriever ======================
sub = jobs[is_open]
cells = sub.groupby(["WindowID", "State"]).indices
pos_in_all = np.flatnonzero(is_open.values)
rows = []
for (w, st), ug in target_users.groupby(["WindowID", "State"]):
    cidx = cells.get((w, st))
    if cidx is None:
        continue
    jidx = pos_in_all[cidx]
    index = faiss.IndexFlatIP(J.shape[1])
    index.add(np.ascontiguousarray(J[jidx], dtype="float32"))
    q = np.ascontiguousarray(U[ug.index.to_numpy()], dtype="float32")
    scores, pos = index.search(q, min(K_RETRIEVE, len(jidx)))
    for uid, s_row, p_row in zip(ug["UserID"].values, scores, pos):
        for rank, (s, pp) in enumerate(zip(s_row, p_row), start=1):
            rows.append((uid, w, job_ids[jidx[pp]], rank, float(s)))
content = pd.DataFrame(rows, columns=["UserID", "WindowID", "JobID", "rank", "score"])
save_and_eval(finalize(content), "content")

# ====================== Collaborative retriever (item-item) ======================
u_codes, u_uniques = pd.factorize(pd.MultiIndex.from_frame(hist[["UserID", "WindowID"]]))
i_codes, i_uniques = pd.factorize(pd.MultiIndex.from_frame(hist[["JobID", "WindowID"]]))
user_lookup = {k: i for i, k in enumerate(u_uniques)}

R = sp.csr_matrix((np.ones(len(hist), dtype=np.float32), (u_codes, i_codes)),
                  shape=(len(u_uniques), len(i_uniques)))
pop = np.asarray(R.sum(axis=0)).ravel()
Rn = (R @ sp.diags(1.0 / np.sqrt(pop))).tocsr()
RnT = Rn.T.tocsr()

meta = jobs[["JobID", "WindowID", "State", "is_open"]].drop_duplicates(["JobID", "WindowID"])
item_df = pd.DataFrame(list(i_uniques), columns=["JobID", "WindowID"]).merge(
    meta, on=["JobID", "WindowID"], how="left")
item_job = item_df["JobID"].values
item_window = item_df["WindowID"].values
item_open = item_df["is_open"].fillna(False).astype(bool).values
item_state = item_df["State"].fillna("").values

cands = [(u, w, st, user_lookup[(u, w)])
         for u, w, st in zip(target_users["UserID"], target_users["WindowID"], target_users["State"])
         if (u, w) in user_lookup]
print("target users with history:", len(cands), "of", len(target_users))

rows = []
for s in range(0, len(cands), CHUNK):
    part = cands[s:s + CHUNK]
    H = R[[c[3] for c in part]]
    scores = ((H @ RnT) @ Rn).tocsr()
    for r, (u, w, st, uc) in enumerate(part):
        a, b = scores.indptr[r], scores.indptr[r + 1]
        idx, sc = scores.indices[a:b], scores.data[a:b]
        keep = item_open[idx] & (item_window[idx] == w) & (item_state[idx] == st)
        keep &= ~np.isin(idx, R.indices[R.indptr[uc]:R.indptr[uc + 1]])
        idx, sc = idx[keep], sc[keep]
        if len(idx) == 0:
            continue
        for rank, t in enumerate(np.argsort(-sc)[:K], start=1):
            rows.append((u, w, item_job[idx[t]], rank, float(sc[t])))
collab = pd.DataFrame(rows, columns=["UserID", "WindowID", "JobID", "rank", "score"])
save_and_eval(collab, "collab")
