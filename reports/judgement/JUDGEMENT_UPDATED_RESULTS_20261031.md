# TalentMatch — Updated Judgement Evaluation Results

## Evaluation Setup

- CV profiles: 30
- Unique CV-job pairs: 452
- Recommendation configurations: 2
- Recommendations per configuration: 300
- Relevance scale: 0 (Not Relevant), 1 (Partially Relevant), 2 (Highly Relevant)
- Final scores available: 451 of 452
- Unresolved item: ITEM-0156

## Evaluation Sources

- `eval/judgement_set.csv`: consolidated final relevance scores.
- `results/judgement/review_round1/internal/talentmatch_matched_skills_final.csv`: recommendation identities and ranks.
- `results/judgement/final_score_metrics_20261031.csv`: updated recommendation metrics.
- `results/judgement/final_decision_provenance.csv`: provenance of the latest adjudication decisions.

The original reviewer scores and consolidated final scores are different scoring versions and must not be mixed.

## Original Reviewer Agreement

| Metric | Result |
|---|---:|
| Evaluated pairs | 452 |
| Exact agreement | 90.93% |
| Cohen's Kappa | 0.8433 |

These values are retained from the previous report and refer to the original reviewer scoring version, not the consolidated final scores.

## Recommendation Quality — Consolidated Final Scores

Primary policy: the unresolved recommendation receives zero gain for metric calculation only. Its final score remains missing in the judgement dataset.

| Metric | pop_weight=0.0 | pop_weight=1.0 |
|---|---:|---:|
| Mean Relevance@10 | 0.8900 | 0.4833 |
| Precision@3 (score >= 1) | 81.11% | 52.22% |
| Precision@10 (score >= 1) | 73.33% | 38.67% |
| Strict Precision@10 (score = 2) | 15.67% | 9.67% |
| NDCG@10 | 0.8205 | 0.4900 |
| Judged share | 99.67% | 100.00% |

## Sensitivity Analysis

When the unresolved item is excluded from the denominator of Precision@10 and mean relevance:

| Metric | pop_weight=0.0 | pop_weight=1.0 |
|---|---:|---:|
| Mean Relevance | 0.8926 | 0.4833 |
| Precision@10 | 73.56% | 38.67% |

NDCG@10 retains the original Top-10 positions and treats the unresolved item's gain as zero in both policies.

## Metric Definitions

- **Mean Relevance@10:** Average relevance grade (0–2) among the top 10 recommendations, averaged equally across 30 CVs.
- **Precision@3:** Proportion of the top 3 recommendations with relevance grade >= 1, averaged equally across 30 CVs.
- **Precision@10:** Proportion of the top 10 recommendations with relevance grade >= 1, averaged across CVs.
- **Strict Precision@10:** Proportion of the top 10 recommendations with relevance grade = 2, averaged across CVs.
- **NDCG@10:** Normalized discounted cumulative gain using linear relevance gains (0, 1, 2), logarithmic rank discounts, and an ideal ranking constructed from the available judged pool for each CV.
- **Judged Share:** Proportion of recommendations with a recorded final relevance score.

## Findings

The configuration without the popularity contribution (`pop_weight=0.0`) outperformed `pop_weight=1.0` on all reported recommendation-quality metrics.

The Precision@10 difference is 34.67 percentage points, and the NDCG@10 difference is 0.3305.

These results describe the tested configurations and do not independently establish why the performance difference occurred.

## Evaluation Status and Limitations

- The consolidated judgement set contains 451 populated final scores and one unresolved case.
- Not every populated final score represents independent human adjudication; some are provisional reviewer agreements or AI-assisted proposals approved by the user.
- ITEM-0156 remains unresolved and is not assigned a final relevance grade.
- The ideal ranking for NDCG@10 is constructed from the judged candidate-job pool available for each CV, rather than all potentially relevant jobs in the full dataset.
- The reported results cover the two retrieval configurations, not a separately evaluated re-ranker.
- The original reviewer agreement metrics and updated recommendation metrics use different scoring versions.
- These are offline evaluation results and do not establish production performance.

## Conclusion

Under the consolidated scoring version, `pop_weight=0.0` achieved better recommendation relevance and ranking quality than `pop_weight=1.0`. The conclusion is limited to the evaluated 30 CV profiles and the available judgement set.
