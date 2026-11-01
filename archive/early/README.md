# Early scripts (superseded)

Smoke tests and first baselines written before the final pipeline. They are kept for history and are not used for any reported result.

- `train_reranker.py`, `compare_lr.py`, `e4_negatives.py`, `e5_diversity.py`: smoke tests on a validation sample. The real versions are `src/train_reranker_v2.py`, `src/compare_lr_real.py`, `src/e4_real.py` and `src/e5_diversity_v1.py`.
- `baseline_popular.py`, `baseline_popular_state.py`, `baseline_tfidf.py`, `popular_jobs.py`: first baselines without the open-jobs filter. The reported baselines are `src/baseline_popular_state_open.py` and `src/baseline_tfidf_period.py`.
