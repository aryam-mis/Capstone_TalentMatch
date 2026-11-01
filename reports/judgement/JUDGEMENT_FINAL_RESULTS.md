# TalentMatch — Final Judgement Evaluation Results

This file extends `JUDGEMENT_UPDATED_RESULTS_20261031.md` with the LightGBM re-ranker, graded in a second round.

## Evaluation Setup

- CV profiles: 30
- Configurations: content only (`pop_weight=0.0`), content with popularity (`pop_weight=1.0`), LightGBM re-ranker
- Recommendations per configuration: 300 (top 10 for each CV)
- Relevance scale: 0 (not relevant), 1 (partly relevant), 2 (highly relevant)
- Round 1: 452 distinct pairs for the two retrieval configurations (451 final scores; ITEM-0156 unresolved)
- Round 2: the re-ranker top 10, 300 pairs = 114 already graded in round 1 + 186 new; all 300 have a final score

## Sources

- `eval/judgement_set.csv`: round 1 final scores.
- `eval/judgement_reranker.csv`: the re-ranker top 10 with final grades (no job texts).
- `eval/judgement_reranker_per_cv.csv`: re-ranker metrics per CV.
- `results/judgement/TalentMatch_Reranker_Final_300_Evaluation.csv`: merged 300 grades (114 from round 1, 186 from round 2).
- `results/judgement/reranker_top10.csv`: the re-ranker recommendations that were graded.

## Grading procedure

Because of the time available, the grades in both rounds were AI-assisted drafts, and the final scores were approved by a team member. The agreement between the round 1 drafts (90.93%, Cohen's kappa 0.843) describes those drafts and is not independent human agreement. No agreement figure is reported for round 2.

## Re-ranker final grades

| Grade | Count |
|---|---:|
| 0 (not relevant) | 149 |
| 1 (partly relevant) | 99 |
| 2 (highly relevant) | 52 |
| Total | 300 |

## Recommendation quality (final scores, 30 CVs, equal weight per CV)

| Metric | Content only | Content with popularity | LightGBM re-ranker |
|---|---:|---:|---:|
| Mean relevance@10 | 0.8900 | 0.4833 | 0.6767 |
| Precision@3 (score >= 1) | 81.11% | 52.22% | 56.67% |
| Precision@10 (score >= 1) | 73.33% | 38.67% | 50.33% |
| Strict Precision@10 (score = 2) | 15.67% | 9.67% | 17.33% |
| NDCG@10 | 0.8205 | 0.4900 | 0.5290* |
| MAP@10 | – | – | 0.4591 |

95% intervals for the re-ranker (bootstrap over the 30 CVs, 2,000 resamples, seed 42): mean relevance 0.500 to 0.877, Precision@3 43.3% to 70.0%, Precision@10 38.7% to 62.0%, strict Precision@10 9.0% to 26.7%, NDCG@10 0.424 to 0.632, MAP@10 0.340 to 0.590.

\* The re-ranker's NDCG@10 uses an ideal ranking built from all graded jobs of each CV (round 1 and its own 186 new pairs), a larger pool than the one used for the other two columns, so it is not strictly comparable. MAP@10 counts grades 1 and 2 as relevant, with the denominator min(relevant jobs in the judged pool, 10).

## Findings

- Content only remains the most relevant configuration in human judgement.
- The re-ranker beats content with popularity on every measure and has the highest share of highly relevant jobs (17.3%).
- Offline (test period), the re-ranker is the best method (NDCG@10 0.080 against 0.030 for content). In human judgement it is less relevant than content only. The re-ranker learned from applications that popular local jobs get applied to; this improves offline recall but not relevance to the profile. This is the gap between logs and preference that the brief warns about, and the reason the app also offers Profile match for cold-start users.

## Limitations

- The grades are AI-assisted drafts approved by team members, not independent double grading.
- The judged pool for each CV covers only the recommended jobs, not every relevant job in the data.
- 30 CVs: the intervals above are wide.
- These are offline results and do not establish production performance.
