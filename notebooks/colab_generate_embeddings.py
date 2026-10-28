# Reproduces artifacts/job_union_bge_small_title_req.npy and artifacts/user_emb_bge_small.npy.
# Run in Google Colab with a T4 GPU, one "# %%" block per cell.
# Inputs are created locally by src/prepare_union.py and src/prepare_for_colab.py
# and uploaded to Google Drive in a folder named talentmatch.

# %% Setup (Colab only: run "!pip -q install sentence-transformers pyarrow" in a cell first)
from google.colab import drive
drive.mount("/content/drive")

# %% Encode jobs and users with the same model and settings
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = "/content/drive/MyDrive/talentmatch"
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
model.half()                  # fp16 on the GPU
model.max_seq_length = 128    # longer texts are truncated to 128 tokens


def encode(texts):
    e = model.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    return e.astype("float16")   # L2-normalised, no query or passage prefix


jobs = pd.read_parquet(f"{BASE}/jobs_open_union.parquet")      # row order must not change
ej = encode(jobs["txt_title_req"].tolist())
assert len(ej) == len(jobs)
np.save(f"{BASE}/job_union_bge_small_title_req.npy", ej)

users = pd.read_parquet(f"{BASE}/users_for_emb.parquet")
eu = encode(users["profile_text"].tolist())
assert len(eu) == len(users)
np.save(f"{BASE}/user_emb_bge_small.npy", eu)
users[["UserID", "WindowID", "State"]].to_parquet(f"{BASE}/users_keys.parquet", index=False)
print("jobs:", ej.shape, "users:", eu.shape)