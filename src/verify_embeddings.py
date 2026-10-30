import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

jobs = pd.read_parquet("artifacts/jobs_open_union.parquet").reset_index(drop=True)
J = np.load("artifacts/job_union_bge_small_title_req.npy", mmap_mode="r")
print("parquet rows:", len(jobs), "| embeddings:", J.shape, J.dtype)
assert len(jobs) == len(J), "row counts differ"

norms = np.linalg.norm(np.asarray(J[:1000], dtype="float32"), axis=1)
print("norm of the first 1000 rows (should be close to 1):",
      round(float(norms.min()), 3), "to", round(float(norms.max()), 3))

rng = np.random.default_rng(args.seed)
idx = np.sort(rng.choice(len(jobs), size=args.n, replace=False))

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
model.max_seq_length = 128
fresh = model.encode(jobs["txt_title_req"].iloc[idx].tolist(), batch_size=32,
                     normalize_embeddings=True, convert_to_numpy=True).astype("float32")

stored = np.asarray(J[idx], dtype="float32")
nxt = np.asarray(J[(idx + 1) % len(jobs)], dtype="float32")
same = (fresh * stored).sum(axis=1)       # same row: should be close to 1
other = (fresh * nxt).sum(axis=1)         # neighbouring row: control, should be clearly lower

print(f"cosine(fresh, stored same row)     min={same.min():.4f} mean={same.mean():.4f}")
print(f"cosine(fresh, stored next row)     mean={other.mean():.4f}")
print("VERDICT:", "OK, order and model match" if same.min() > 0.98 else "MISMATCH, do not use these files")