# Artifacts

## Committed

- `reranker_v2/model.txt`: LightGBM LambdaRank model (best iteration 116, seed 42).
- `reranker_v2/meta.json`: version, feature order, parameters, data periods, SHA-256 of the model, candidate sizes, embedding model, and validation/test results.

Load the model and rank jobs for a new profile in a fresh process (needs the large files below):

```
python src/profile_recommender.py --state CA --degree "Bachelor's" --major "Business Administration" --years 5 --roles "Sales Associate"
```

## Not committed (large)

Shared on Drive as `artifacts_for_aryam.zip`; unzip into `artifacts/`:

- `jobs_open_union.parquet` and `job_union_bge_small_title_req.npy`: open jobs and their embeddings (BAAI/bge-small-en-v1.5).
- `users_keys.parquet` and `user_emb_bge_small.npy`: user keys and embeddings.

Built locally:

- `data/processed/jobs_text_open.parquet` (job requirements and descriptions for the Why-this-job explanations): `python src/build_jobs_text.py`