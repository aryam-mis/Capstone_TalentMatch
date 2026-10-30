# TalentMatch: Two-Sided Job and Candidate Recommendation Engine

Olo Academy AI Engineering Bootcamp (AIEP), Module 8 capstone, Track C (HR tech, recommendation systems).
Team: Aryam, Layan, Amnia.

## Problem and motivation

Job platforms receive thousands of applications per posting. Keyword filters reject good candidates who describe their experience differently, and job seekers scroll through postings that do not fit. TalentMatch ranks jobs for a candidate and candidates for a job, works on a profile or a job typed in a few seconds with no application history, and shows why each job appears. It ranks and explains; the recruiter decides.

- **Job seeker view:** type degree, major, years, past roles, skills, state and city, and get ten ranked jobs with the matched skills and the sentence that supports each one.
- **Recruiter view:** type a job (or pick one) and get ten ranked candidates by content similarity.

## Architecture

```mermaid
flowchart LR
  P[Typed profile] --> C[Content retriever: bge-small, top 100]
  P --> B[Collaborative: item-item, users with history]
  P --> O[State popularity: top 50]
  C --> M[Merge and 12 features]
  B --> M
  O --> M
  M --> R[LightGBM LambdaRank re-ranker]
  R --> T[Top 10 and Why this job]
  J[Typed job] --> U[User index: bge-small, window and state] --> K[Top 10 candidates, content only]
```

## Data

- **Source:** CareerBuilder Job Recommendation Challenge on Kaggle (users, jobs, applications, work history), as specified in the capstone brief. The brief gives no licence text. Skills vocabulary: ESCO v1.2.1 (`skills_en.csv`).
- **Not in this repository:** raw data, embeddings and large tables. Download the Kaggle files and unzip them under `data/` (scripts expect `data/apps/apps.tsv`; the `clean_*.py` scripts in `src/` write `data/processed/`). The large model-side files are listed in `artifacts/README.md`.
- **Time-based split** (`src/make_split.py`): train up to 2012-06-05 (1,290,087 applications), validation 2012-06-06 to 2012-06-13 (154,298), test 2012-06-18 to 2012-06-26 (153,355). The re-ranker is trained on the last week of the train period (2012-05-30 to 2012-06-05), early-stopped on validation and judged once on test. Decisions frozen before the test run are in `docs/frozen_before_test.md`.

## Install

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Developed with Python 3.14 and LightGBM 4.7.0 on Windows; no GPU needed.

## Quick checks (no data needed)

```
python -m pytest tests -q
```

This runs the smoke tests, including loading `artifacts/reranker_v2` (checksum, load, ranking of cold-start rows).

## Run the demo

Needs the large files from `artifacts/README.md` and `python src/build_jobs_text.py` once.

```
python -m streamlit run app/app.py
```

Rank jobs for a new profile from the command line, loading the artifact in a fresh process:

```
python src/profile_recommender.py --state CA --degree "Bachelor's" --major "Business Administration" --years 5 --roles "Sales Associate"
```

## Reproduce the main results

```
# training period (2012-05-30 to 2012-06-05)
python src/gen_candidates.py --name train_rr --start 2012-05-30 --end 2012-06-06 --history_before 2012-05-30 --open_from 2012-05-30 --k 100
python src/gen_pop_candidates.py --users_from results/recs/train_rr_content.csv --open_from 2012-05-30 --end 2012-06-06 --history_before 2012-05-30 --k 50 --out results/recs/train_rr_pop.csv
python src/build_features.py --content results/recs/train_rr_content.csv --collab results/recs/train_rr_collab.csv --pop results/recs/train_rr_pop.csv --labels results/recs/train_rr_labels.csv --pop-before 2012-05-30 --period-start 2012-05-30 --out data/processed/features_train_rr3.parquet

# validation (2012-06-06 to 2012-06-13)
python src/gen_candidates.py --name val_check --start 2012-06-06 --end 2012-06-14 --history_before 2012-06-06 --open_from 2012-06-06 --k 100 --eval_split val
python src/gen_pop_candidates.py --users_from results/recs/val_check_content.csv --open_from 2012-06-06 --end 2012-06-14 --history_before 2012-06-06 --k 50 --out results/recs/val_pop.csv
python src/build_features.py --content results/recs/val_check_content.csv --collab results/recs/val_check_collab.csv --pop results/recs/val_pop.csv --labels results/recs/val_check_labels.csv --pop-before 2012-06-06 --period-start 2012-06-06 --out data/processed/features_val3.parquet

# train, then evaluate on validation
python src/train_reranker_v2.py --train data/processed/features_train_rr3.parquet --val data/processed/features_val3.parquet
python src/evaluate.py --recs results/recs/rerank_v2.csv --split val --name rerank_v2


# test (open jobs from 2012-06-18), model and weights frozen
python src/gen_candidates.py --name test --start 2012-06-14 --end 2012-06-27 --history_before 2012-06-14 --open_from 2012-06-18 --k 100 --eval_split test
python src/gen_pop_candidates.py --users_from results/recs/test_content.csv --open_from 2012-06-18 --end 2012-06-27 --history_before 2012-06-14 --k 50 --out results/recs/test_pop.csv
python src/build_features.py --content results/recs/test_content.csv --collab results/recs/test_collab.csv --pop results/recs/test_pop.csv --labels results/recs/test_labels.csv --pop-before 2012-06-14 --period-start 2012-06-18 --out data/processed/features_test3.parquet
python src/e5_diversity_v1.py --artifact artifacts/reranker_v2 --features data/processed/features_test3.parquet --lambdas 0 0.15 --prefix results/recs/test_rerank_v2_lam
python src/evaluate.py --recs results/recs/test_rerank_v2_lam0.csv --split test --name test_rerank_v2_lam0
```

Other scripts: `src/rrf_hybrid.py` (hybrid), `src/baseline_tfidf_period.py` (TF-IDF), `src/compare_lr_real.py` (Logistic Regression), `src/make_random_negs.py`, `src/add_cos.py` and `src/e4_real.py` (E4), `src/e5_popularity.py` and `src/e5_diversity_v1.py` (E5). Seeds are fixed to 42.

## Results (test period, `evaluate.py`, 47,214 users)

| Method | P@10 | NDCG@10 | MAP@10 | Recall@100 |
| --- | --- | --- | --- | --- |
| Collaborative (item-item) | 0.0108 | 0.0274 | 0.0174 | 0.1035 |
| Content (bge-small) | 0.0095 | 0.0300 | 0.0196 | 0.1661 |
| TF-IDF | 0.0109 | 0.0340 | 0.0218 | 0.1864 |
| Popularity by state | 0.0177 | 0.0360 | 0.0211 | 0.2196 |
| Hybrid RRF | 0.0157 | 0.0466 | 0.0302 | 0.2309 |
| Hybrid RRF3 | 0.0246 | 0.0607 | 0.0372 | 0.2844 |
| Logistic Regression | 0.0262 | 0.0631 | 0.0391 | 0.2999 |
| **LightGBM re-ranker** | **0.0304** | **0.0801** | **0.0513** | **0.3060** |

All numbers are in `docs/test_results.csv`. Hard negatives beat random negatives (E4), and a popularity penalty of 0.15 lowers the share of recommendations on the 1% most popular jobs from 18.9% to 14.2% at a 2.2% NDCG@10 cost (E5; `docs/e5_test.txt`, `docs/e5_val_v2.txt`). A typed profile is answered in about 100 ms once warm (1.3 s on the first call).

## Limitations and future work

- One trained model per method, no confidence intervals; part of the validation-to-test gap is validation optimism.
- The data is US applications from April to June 2012 in seven independent windows, not the Saudi market.
- Popularity bias: the re-ranker favours popular jobs; the diversity weight only reduces it.
- The skill matcher (ESCO, FlashText) is conservative and its precision is not measured.
- The recruiter view is content only, and the human judgement set is still open.
- Next: confidence intervals, cold and warm users evaluated separately, richer profile text, diversity-aware training.

## Responsible AI

Hiring is high-stakes: the system ranks and explains and never rejects, scores are comparable only inside one list, city and missing profile fields can act as proxies, and any CV used in a demo must be anonymised. See the report for details.

## Repository layout

`src/` code, `app/` Streamlit app, `tests/` smoke tests, `artifacts/` model and its notes, `docs/` result tables and frozen decisions. `data/` and `results/` are not versioned.
