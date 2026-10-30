# Decisions frozen before running anything on test

- Model: artifacts/reranker_v2, trained on the training period (2012-05-30 to 06-05), early-stopped on val (best iteration 116). No retraining on val.
- Features: content_score, content_rank, collab_score, collab_rank, pop_rank, same_city, job_pop, job_age_days, days_left, user_n_hist, TotalYearsExperience, WorkHistoryCount.
- Diversity weight: lambda = 0.15 (largest lambda losing under 2% of NDCG@10 on val). Lambda = 0 is reported as well.
- Candidates: content and collab (k=100 each) plus state popularity (k=50) per user.
- Test open-jobs filter: EndDate >= 2012-06-18 and StartDate < 2012-06-27. History and popularity: all applications before 2012-06-14.
- Metrics: evaluate.py (P@10, NDCG@10, MAP@10, Recall@50, Recall@100), E5 coverage and top-1% share.
- Test is run once per method. No tuning on test.