# TalentMatch — Cold-Start Ranking Experiment

## Objective
Improve job recommendations for users with limited or no interaction history by considering profile relevance, experience requirements, and preferred city.

## Approach
- Increased content candidate pool to 500 jobs.
- Used BGE embedding similarity for ranking.
- Added a penalty for senior-level job titles for users with less than 2 years of experience.
- Added an experimental penalty for detected experience requirements exceeding the user's experience.
- Added a city bonus of 0.05 when content similarity is at least 0.68.
- Precomputed experience metadata in `artifacts/cold_start_experience.parquet`.

## Evaluation
Six synthetic profiles were tested across AI, Accounting, Computer Science, Data Science, Marketing, and Software Engineering.

Each configuration generated 10 recommendations per profile.

### City-match results

| Configuration | City matches |
|---|---|
| Original recommender | 54/60 (90.0%) |
| Experimental, before experience adjustment | 23/60 (38.3%) |
| Experimental, experience adjustment, city bonus 0.03 | 20/60 (33.3%) |
| Experimental, experience adjustment, city bonus 0.08 | 46/60 (76.7%) |
| Experimental, experience adjustment, city bonus 0.05 with similarity threshold 0.68 | 32/60 (53.3%) |

The original recommender achieved higher city matching but produced several visibly off-topic recommendations for the AI test profile. The experimental configuration trades some geographic matching for greater emphasis on content similarity.

City matching alone is not a measure of overall recommendation relevance.

## Performance
- Experimental initialization before experience caching: 76.8 seconds.
- Experimental initialization after caching: 18.4 seconds.
- Reduction in observed initialization time: approximately 76%.
- All 120 recommendations across original and experimental runs preserved their ranking after the caching optimization.

These timings are from individual local runs, not a repeated benchmark.

## Limitations
- The evaluation uses six synthetic profiles, not real-user satisfaction data.
- Experience extraction relies on heuristic patterns and may miss requirements or detect years in unrelated contexts.
- Some recommended jobs still require manual relevance review.
- The city bonus and similarity threshold are experimental hyperparameters.
- The historical CareerBuilder dataset does not represent current job vacancies.
- The experimental ranking is not yet validated as superior to the original model.

## Status
Experimental implementation complete and pushed to branch `experiment/cold-start-ranking`.

Pending:
1. Human review of recommendation relevance and experience suitability.
2. Team code review.
3. Approval before integration into the main recommender.
