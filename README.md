
# TalentMatch
### Two-Sided Job and Candidate Recommendation Engine

**AI Engineering Bootcamp Capstone | Olo Academy**

## Project Overview

TalentMatch is an AI-powered recommendation system designed to improve the matching process between job seekers and job opportunities.

The project combines semantic text representations, structured candidate and job information, and skills-based matching to support more relevant recommendations.

## Project Objectives

- Match candidate profiles with relevant job opportunities.
- Use semantic embeddings to understand job descriptions and candidate qualifications.
- Extract and normalize job-related skills using the ESCO taxonomy.
- Evaluate recommendation quality using offline metrics and human relevance judgments.
- Explore ranking improvements and cold-start recommendations.

## Data Sources

| Dataset | Purpose |
|---|---|
| CareerBuilder | Job postings, user profiles, applications, and interaction history |
| ESCO | Standardized occupational skills and terminology |
| Synthetic CVs | Candidate profiles for recommendation evaluation |

The project uses 30 synthetic CV profiles for the initial judgment-set evaluation.

## System Workflow

1. **Data Understanding and Preparation**
   Explore datasets, inspect missing values and duplicates, clean job descriptions, and prepare text for downstream processing.

2. **Semantic Retrieval**
   Represent job postings and candidate profiles using BGE-small embeddings and retrieve potentially relevant jobs.

3. **ESCO Skill Extraction**
   Identify job-related skills from descriptions and requirements using ESCO terminology.

4. **Recommendation and Ranking**
   Combine candidate-job relevance signals to improve recommendation ordering.

5. **Evaluation**
   Evaluate recommendations using offline metrics and independent human relevance judgments.

## ESCO Skill Extraction

The ESCO extraction component focuses on identifying relevant skills from job titles, descriptions, and requirements.

Current improvements include:

- Improved recognition of responsibility headings.
- Better handling of job descriptions containing company introductions.
- More targeted extraction of job-role content.
- Internal tests for description and requirements extraction.

**Key files:**

- `src/extract_esco_skills_candidate.py` — ESCO extraction implementation.
- `src/create_esco_review_sample.py` — Review sample preparation.
- `reports/esco/esco_final_eval_40_review.csv` — Preliminary review of 40 job postings.

The 40-job review is an initial quality assessment and should not be interpreted as a fully validated human-annotated benchmark.

### Run Internal Tests

```bash
python src/extract_esco_skills_candidate.py --self-test
```

## Recommendation Evaluation

The evaluation design includes:

- 30 synthetic candidate profiles.
- Top-10 recommendation lists generated using two retrieval configurations.
- 452 unique candidate-job pairs collected for review.
- Two independent reviewers assigning relevance scores:
  - `0` — Not relevant
  - `1` — Partially relevant
  - `2` — Relevant

Planned evaluation measures include:

- Precision@3
- NDCG@10
- Inter-rater agreement

**Status:** Independent human annotation and final evaluation metrics are pending.

## Repository Structure

```text
Capstone_TalentMatch/
├── notebooks/
│   └── 01_data_understanding.ipynb
├── src/
│   ├── extract_esco_skills_candidate.py
│   └── create_esco_review_sample.py
├── reports/
│   └── esco/
│       └── esco_final_eval_40_review.csv
└── README.md
```

This structure highlights selected components; it is not an exhaustive repository listing.

## Technologies

Python, Pandas, NumPy, Sentence Transformers, BGE-small, FAISS, ESCO, and Jupyter Notebook.

## Project Status

**In development.**

Data exploration, semantic retrieval experiments, and ESCO extraction improvements have been implemented. Human relevance evaluation, integration, and final pipeline validation remain in progress.

## Team

Developed as a collaborative capstone project for the Olo Academy AI Engineering Bootcamp.
