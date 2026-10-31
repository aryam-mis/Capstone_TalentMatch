import argparse
import os

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from split_utils import load_apps, temporal_split

p = argparse.ArgumentParser()
p.add_argument("--cvs", required=True, help="CSV with columns: cv_id, state, profile_text")
p.add_argument("--out", default="results/judgement/judgement_template.csv")
p.add_argument("--period_start", default="2012-06-06")
p.add_argument("--period_end", default="2012-06-14")
p.add_argument("--k", type=int, default=10)
p.add_argument("--pop_weight", type=float, default=0.0,
               help="weight of the popularity ranking inside RRF (0 = content only)")
args = p.parse_args()

START, END = pd.Timestamp(args.period_start), pd.Timestamp(args.period_end)
RRF_K, POOL = 60, 100
os.makedirs(os.path.dirname(args.out), exist_ok=True)

cvs = pd.read_csv(args.cvs)
cvs["state"] = cvs["state"].astype(str).str.strip().str.upper()

# Jobs open in the period; row i of the dataframe matches row i of the embeddings
jobs = pd.read_parquet("artifacts/jobs_open_union.parquet").reset_index(drop=True)
J = np.load("artifacts/job_union_bge_small_title_req.npy", mmap_mode="r")
assert len(J) == len(jobs), (len(J), len(jobs))
is_open = ((jobs["EndDate"].isna() | (jobs["EndDate"] >= START)) &
           (jobs["StartDate"].isna() | (jobs["StartDate"] < END)) &
           ~jobs.duplicated(["JobID", "WindowID"]))
open_pos = np.flatnonzero(is_open.values)

# State popularity from TRAIN applications only
train, _, _ = temporal_split(load_apps())
pop = train.groupby(["JobID", "WindowID"]).size().rename("pop").reset_index()
jobs = jobs.merge(pop, on=["JobID", "WindowID"], how="left").fillna({"pop": 0})
assert len(jobs) == len(J)

# Encode CV profiles with the same model and settings as the user embeddings
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
model.max_seq_length = 128
Q = model.encode(cvs["profile_text"].str.lower().tolist(), batch_size=32,
                 normalize_embeddings=True, convert_to_numpy=True).astype("float32")

by_state = jobs.iloc[open_pos].groupby("State").indices
rows = []
for i, (cv_id, st) in enumerate(zip(cvs["cv_id"], cvs["state"])):
    cidx = by_state.get(st)
    if cidx is None:
        print("no open jobs for state", st, "-> cv", cv_id)
        continue
    pos = open_pos[cidx]
    index = faiss.IndexFlatIP(J.shape[1])
    index.add(np.ascontiguousarray(J[pos], dtype="float32"))
    scores, nn = index.search(Q[i:i + 1], min(POOL, len(pos)))
    content_rank = {int(pos[n]): r for r, n in enumerate(nn[0], start=1)}
    pop_order = pos[np.argsort(-jobs["pop"].values[pos], kind="stable")][:POOL]
    pop_rank = {int(x): r for r, x in enumerate(pop_order, start=1)}

    # Reciprocal rank fusion with a weight on the popularity list
    fused = {}
    for ranks, weight in ((content_rank, 1.0), (pop_rank, args.pop_weight)):
        for j, r in ranks.items():
            fused[j] = fused.get(j, 0.0) + weight / (RRF_K + r)
    top = sorted(fused.items(), key=lambda kv: -kv[1])[:args.k]

    for rank, (j, s) in enumerate(top, start=1):
        rows.append({"cv_id": cv_id, "state": st, "rank": rank,
                     "JobID": jobs["JobID"].values[j], "WindowID": jobs["WindowID"].values[j],
                     "score_rrf": round(s, 5),
                     "in_content_top100": j in content_rank, "in_pop_top100": j in pop_rank,
                     "job_text": str(jobs["txt_title_req"].values[j])[:300],
                     "grader1": "", "grader2": ""})

out = pd.DataFrame(rows)
out.to_csv(args.out, index=False, encoding="utf-8-sig")
print("saved", args.out, len(out), "rows for", out["cv_id"].nunique(), "CVs",
      "| pop_weight =", args.pop_weight)