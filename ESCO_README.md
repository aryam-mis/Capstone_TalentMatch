# TalentMatch — ESCO Skills Extraction

## 1. Overview

This module extracts standardized skills from CareerBuilder job advertisements using **ESCO v1.2.1** and **FlashText**.

The extracted skills can support job recommendation explanations and future skill-matching features.

## 2. Data Preparation

The extraction pipeline uses the cleaned CareerBuilder jobs dataset.

**Input:** `data/processed/jobs_clean.tsv`

The cleaned jobs dataset contains 1,092,096 job records across the available time windows.

The ESCO taxonomy is loaded from the local `skills_en.csv` file.

## 3. Implementation

**Main script:** `src/extract_esco_skills_candidate.py`

**Libraries:** pandas, flashtext

The implementation includes:

- ESCO preferred labels and alternative labels.
- Filtering ambiguous skill names and aliases.
- Context rules to reduce false positives.
- Extraction from job requirements and responsibilities.
- Filtering non-job content and promotional sections.
- Handling unsectioned descriptions.
- Chunk-based processing for large datasets.
- Mapping extracted skills to ESCO concept URIs.
- Review flags for cases requiring additional attention.

## 4. Improvements

Several experimental versions were developed to improve extraction quality.

The current candidate version includes fixes for:

- Incorrect mappings caused by general phrases.
- Missing job responsibilities after benefits sections.
- Figurative expressions such as "the fine art of customer service."
- Selected degree-major mentions incorrectly treated as job skills.
- Ambiguous ESCO aliases.

These improvements were tested on selected job advertisements. Comprehensive independent quality evaluation remains pending.

## 5. Running the Pipeline

**Window 1:**

`python src/extract_esco_skills_candidate.py --full --windows 1 --output data/processed/jobs_esco_window1_final.tsv`

**Windows 1 and 2:**

`python src/extract_esco_skills_candidate.py --full --windows 1,2 --output data/processed/jobs_esco_windows_1_2.tsv`

The second command is available if the team decides to use both windows.

## 6. Final Window 1 Results

Final Window 1 results:

| Metric | Result |
|---|---:|
| Total jobs | 285,155 |
| Jobs with extracted skills | 242,185 |
| Jobs without extracted skills | 42,970 |
| Extraction coverage | 84.9% |
| Duplicate JobID–WindowID pairs | 0 |
| Skills–URI count mismatches | 0 |


## 7. Output Structure

The full extraction output includes:

- `JobID`
- `WindowID`
- `Title`
- `TextSource`
- `ExtractedSkills`
- `ESCO_URIs`
- `MatchedLabels`
- `ReviewFlag`

The output can be joined with job data using `JobID` and `WindowID`.

## 8. Evaluation and Limitations

Coverage is not accuracy. Precision and recall require manual evaluation.
