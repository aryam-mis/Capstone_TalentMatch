# Capstone Project Report

**Project Title:** Talent Match 

**Student Name:** Aryam Almohammadi, Layan Almohammadi, Omnia Alsadi

**Cohort:** 

**Date:** 31 Oct 2026

**Repository URL:** [GitHub repository link]

---

## 1. The Problem

### 1.1 Problem Definition

[Describe the problem your project addresses in one clear sentence that a non-specialist can understand.]

[Explain what the system predicts, classifies, retrieves, recommends, or generates.]

### 1.2 Intended Users and Decisions

[Identify the intended users of the system.]

[Explain how users will use the system's output and what decisions it will help them make.]

### 1.3 Why This Problem Matters

[Describe the current situation and the challenges associated with existing approaches.]

[Explain what changes when your solution is introduced.]

[Describe the expected benefits, such as improved efficiency, reduced errors, better decision-making, or saved time, where applicable.]

### 1.4 Success Criteria

[Define what a successful solution would look like.]

[Specify a measurable performance target appropriate to your problem.]

[Explain the basis for this target, such as project requirements, a baseline comparison, or a justified objective.]

---

## 2. The Data

### 2.1 Data Source and License

**Dataset Name:** [Enter the dataset name]

**Source:** [Enter the official dataset URL or source]

**License:** [Enter the license or usage terms]

**Date Accessed:** [Enter the date you obtained the data]

[Explain why the dataset is appropriate for the project.]

[Describe any restrictions on data access, use, or redistribution.]

### 2.2 Data Collection and Reproducibility

[Explain how the data were obtained, such as downloading a dataset, using an API, collecting records, or manually labeling examples.]

[Describe the steps another person would need to follow to obtain the same data.]

[Identify any scripts, notebooks, or commands used to collect or prepare the data.]

[If the dataset cannot be redistributed, explain how another person can access it lawfully.]

### 2.3 Dataset Card

#### Purpose and Intended Use

[Describe the purpose of the dataset and the tasks it is intended to support.]

#### Dataset Composition

[Describe the dataset's size, structure, main features, and target variable, where applicable.]

#### Data Collection Process

[Explain how the original data were collected and any relevant sampling methods.]

#### Intended Users

[Identify the people or systems expected to use the dataset.]

#### Out-of-Scope Uses

[Describe uses that the dataset does not adequately support or that could lead to misleading conclusions.]

#### Biases and Limitations

[Describe known or observed biases, coverage gaps, representation issues, and other limitations.]

#### Privacy and Ethical Considerations

[Explain relevant privacy, consent, licensing, and ethical considerations. If no personal data are involved, state this where appropriate.]

### 2.4 Dataset Structure

**Number of Rows:** [Enter the number of rows]

**Number of Columns:** [Enter the number of columns]

**Unit of Observation:** [Explain what one row represents.]

**Feature Columns:** [List the main input variables.]

**Target Column:** [Enter the target variable, if applicable.]

[Explain the meaning of the most important columns and how they contribute to the task.]

### 2.5 Target Variable and Distribution

[Define the target variable and explain how it is represented.]

[For classification, report the number and percentage of examples in each class.]

[For regression, describe the target's range and distribution.]

[For retrieval or recommendation, explain how relevant results or ground-truth labels are defined.]

[Discuss any class imbalance, sparsity, or other distributional challenges.]

### 2.6 Data Quality Issues

[Describe the actual data quality issues identified during the project.]

Consider the following possibilities:

- Missing values.
- Duplicate records.
- Incorrect or inconsistent values.
- Outliers.
- Imbalanced classes.
- Sampling bias or limited coverage.
- Noisy or ambiguous labels.
- Potential data leakage.

For each relevant issue, explain:

1. What the issue was.
2. How it was identified.
3. How it was handled.
4. What the decision may have changed or cost.

[Include a figure showing the target distribution, missing values, or another important data quality finding.]

**Figure 1:** [Enter a descriptive caption for the figure.]

---

## 3. Exploration

### 3.1 Exploration Objective

[Explain what you wanted to understand about the dataset before building the model.]

[Describe the questions explored through data analysis and visualization.]

### 3.2 Key Finding 1

**Figure 2:** [Insert or reference the first important chart.]

**What the Figure Shows:**

[Describe the patterns or relationships visible in the figure.]

**Conclusion:**

[Explain what you learned from this observation.]

**Decision Affected:**

[Explain what you changed in your approach because of this finding.]

### 3.3 Key Finding 2

**Figure 3:** [Insert or reference the second important chart.]

**What the Figure Shows:**

[Describe the patterns or relationships visible in the figure.]

**Conclusion:**

[Explain what you learned from this observation.]

**Decision Affected:**

[Explain how this finding influenced feature selection, preprocessing, model selection, or evaluation.]

### 3.4 Additional Finding (Optional)

[Include a third finding only if it provides meaningful evidence or changed an important project decision.]

**Evidence:**

[Reference the chart or analysis.]

**Conclusion:**

[Explain the finding.]

**Decision Affected:**

[Explain the resulting action.]

---

## 4. Method — What I Built and What I Rejected

### 4.1 End-to-End Pipeline

[Describe the complete system from data ingestion to final prediction, retrieval, or recommendation.]

Cover the relevant stages:

1. Data ingestion.
2. Data validation and cleaning.
3. Preprocessing and feature engineering.
4. Training, validation, and test splitting.
5. Baseline implementation.
6. Model training.
7. Hyperparameter selection, if applicable.
8. Model evaluation.
9. Model or artifact persistence.
10. Inference or deployment, if implemented.

[Explain how these stages connect.]

[Identify the scripts, modules, notebooks, or services responsible for each stage.]

**Figure 4: End-to-End Pipeline**

[Insert a diagram showing the actual project pipeline.]

### 4.2 Preprocessing and Feature Engineering

[Describe the preprocessing steps used in the final implementation.]

For each important transformation, explain:

- What was transformed.
- Why the transformation was necessary.
- How it was implemented.
- Whether it was fitted using training data only.

#### Feature 1: [Feature Name]

[Describe the feature, how it was created or processed, and why it may provide useful information.]

#### Feature 2: [Feature Name]

[Describe the feature, how it was created or processed, and why it may provide useful information.]

#### Additional Features

[Describe other relevant features.]

[If no new features were engineered, explain which original features were retained and why.]

### 4.3 Model Selection

**Selected Model:** [Enter the model or algorithm name]

[Explain why this model was suitable for the problem type, dataset size, feature representation, and available computational resources.]

[Discuss the interpretability required by the intended users.]

[Describe the important model settings or hyperparameters and how they were selected.]

### 4.4 Baseline and Alternative Approaches

**Baseline:** [Describe the simple baseline model or method.]

**Alternative Approach:** [Name another model or method considered or tested.]

[Explain why the baseline was useful and what the alternative offered.]

#### Experimental Evidence

| Approach | Configuration | Evaluation Metric | Result | Decision |
|---|---|---|---|---|
| Baseline | [Details] | [Metric] | [Result] | [Retained or rejected] |
| Alternative Model | [Details] | [Metric] | [Result] | [Retained or rejected] |
| Final Model | [Details] | [Metric] | [Result] | [Selected model] |

[Explain which approaches were actually implemented and tested.]

[If an alternative was rejected, provide the measured evidence and reasoning.]

[Do not report hypothetical experiments as completed experiments.]

### 4.5 Data Leakage Prevention

[Explain how the dataset was split and how information from the test set was prevented from influencing training or model selection.]

[Describe the checks performed for target leakage, duplicated examples across splits, inappropriate features, and preprocessing fitted on the wrong data.]

[Explain how you verified that the final evaluation used data not used to fit the final model.]

[If a particular check was not performed, state this honestly and identify it as work still needed.]

### 4.6 Implementation Challenges and Failed Attempts

[Describe a technical problem or experiment that did not work as expected.]

[Explain the symptoms, your investigation, the likely cause, and the change made in response.]

[If no substantial failed attempt has occurred yet, complete this section when relevant evidence becomes available.]

---

## 5. Results

### 5.1 Evaluation Setup

**Dataset Split:** [Enter the training, validation, and test proportions or describe the evaluation protocol.]

**Evaluation Sample Size:** [Enter the number of test examples.]

**Random Seed:** [Enter the seed, if one was used.]

**Evaluation Metrics:** [List the metrics used.]

[Explain how the test set was kept separate from model fitting and model selection.]

### 5.2 Baseline Performance

**Baseline Method:** [Enter the baseline approach.]

**Baseline Result:** [Enter the measured result.]

[Explain why this baseline is an appropriate reference for the problem.]

### 5.3 Final Model Performance

**Final Model:** [Enter the selected model.]

**Primary Metric:** [Enter the metric and measured result.]

**Additional Metrics:** [Enter other relevant measured results.]

[Explain why the primary metric is appropriate for the project objective, data distribution, and consequences of prediction errors.]

### 5.4 Results Comparison

| Model or Version | Primary Metric | Secondary Metric | Evaluation Notes |
|---|---|---|---|
| Baseline | [Result] | [Result] | [Notes] |
| Initial Model | [Result] | [Result] | [Notes] |
| Improved Model | [Result] | [Result] | [Notes] |
| Final Model | [Result] | [Result] | [Notes] |

[Remove rows that do not apply.]

[Use actual experimental results and evaluate compared approaches under a consistent protocol.]

### 5.5 Visual Evaluation

[Insert a confusion matrix, learning curve, precision-recall curve, residual plot, retrieval evaluation chart, or another figure appropriate to the problem.]

**Figure 5:** [Enter a descriptive caption.]

[Explain the main pattern shown in the figure and how it supports or qualifies the numerical results.]

### 5.6 Practical Interpretation

[Translate the evaluation results into terms that matter to the intended users.]

[Explain what the model does well and where its performance remains insufficient.]

[State whether the model meets the success criteria defined in Section 1.]

[Distinguish measured performance from expected performance in a real deployment.]

---

## 6. Error Analysis

### 6.1 Error Identification

[Explain how you identified incorrect predictions, failed retrievals, poor recommendations, or other task-specific errors.]

[Describe the sample of errors examined and how it was selected.]

### 6.2 Common Error Patterns

#### Pattern 1: [Describe the Observed Pattern]

**Evidence:**

[Provide counts, error rates, examples, or a relevant figure.]

**Possible Explanation:**

[Explain why this pattern may occur.]

#### Pattern 2: [Describe the Observed Pattern]

**Evidence:**

[Provide counts, error rates, examples, or a relevant figure.]

**Possible Explanation:**

[Explain why this pattern may occur.]

[Distinguish recurring patterns from isolated mistakes. Do not claim a pattern unless the analysis supports it.]

### 6.3 Source of Errors

[Assess whether the observed errors appear to arise primarily from the model, the data, the labeling process, or limitations inherent in the problem.]

[Explain what evidence supports your assessment and what remains uncertain.]

### 6.4 Impact on Users and Fairness

[Discuss whether errors have different consequences for different user groups, categories, or operating conditions.]

[If relevant, compare error rates across groups using appropriate measurements and sufficient sample sizes.]

[Explain which errors are most consequential and whether the current evaluation captures their impact.]

[If fairness could not be evaluated because the required data were unavailable, explain this limitation rather than assuming the system is fair.]

---

## 7. Limitations

### 7.1 Main Threat to Validity

**The main reason this result might be wrong is...**

[Complete this statement with the most important limitation supported by your project evidence.]

[Explain why this limitation could make the reported results overly optimistic, unstable, or unrepresentative.]

### 7.2 Data Limitations

[Describe specific limitations of the dataset, such as sample size, missing populations, narrow geographic coverage, limited time periods, noisy labels, or incomplete features.]

[Explain which conclusions the data can support and which conclusions they cannot support.]

### 7.3 Evaluation Limitations

[Explain what your evaluation does and does not establish.]

[Discuss whether the test data represent future or real-world conditions.]

[Identify untested assumptions, external validation requirements, or uncertainty in the reported metrics.]

### 7.4 Deployment Risks

[Describe situations in which the system could fail after deployment.]

Consider relevant issues such as:

- Changes in the data distribution.
- Unexpected input formats.
- Unavailable dependencies.
- Latency or computational constraints.
- Silent performance degradation.
- Changes in user behavior.

[Explain how such failures could be detected or monitored.]

### 7.5 Evidence Needed for Trust

[Specify the additional data, experiments, independent evaluations, monitoring, or expert review required before the system could responsibly support real decisions.]

[For each requirement, explain which uncertainty it would reduce.]

---

## 8. Next Steps

[Prioritize two or three concrete improvements based on the limitations identified in Section 7.]

### Priority 1: [Improvement Title]

**Action:**

[Describe the work to be performed.]

**Reason:**

[Identify the limitation or error pattern this addresses.]

**Estimated Cost:**

[Describe the expected time, data, or computational resources.]

**Expected Benefit:**

[Explain the measurable improvement or uncertainty reduction sought.]

### Priority 2: [Improvement Title]

**Action:**

[Describe the work to be performed.]

**Reason:**

[Identify the limitation or error pattern this addresses.]

**Estimated Cost:**

[Describe the expected time, data, or computational resources.]

**Expected Benefit:**

[Explain the measurable improvement or uncertainty reduction sought.]

### Priority 3 (Optional): [Improvement Title]

[Include only if a third priority is justified.]

**Action:** [Describe the work.]

**Reason:** [Explain why it is needed.]

**Estimated Cost:** [Estimate the resources required.]

**Expected Benefit:** [Describe the intended outcome.]

---

## 9. How to Run It

### 9.1 Requirements

**Operating System:** [Enter the supported operating system.]

**Python Version:** [Enter the tested Python version.]

**Dependencies:** [Reference requirements.txt, pyproject.toml, environment.yml, or the actual dependency specification.]

**Data Requirements:** [Explain how to obtain and configure the required data.]

### 9.2 Installation

[Provide the commands needed to clone the repository, create the environment, and install dependencies.]

```bash
git clone [repository URL]
cd [repository directory]
```

[Add the actual environment creation and activation commands.]

[Add the correct dependency installation command.]

### 9.3 Training

**Training Command:**

```bash
[Insert the actual command used to train the model]
```

[Explain any required arguments, input files, or configuration settings.]

### 9.4 Running the System

**Run Command:**

```bash
[Insert the actual command used to launch the application or inference process]
```

[Explain how to submit an input and where to find the output.]

**Example Input:** [Provide a valid example.]

**Expected Output:** [Describe the output format.]

[If the project is a notebook or offline pipeline rather than a service, document the actual execution procedure instead of inventing an API.]

---

## References

[Include every dataset, paper, documentation page, library, and external resource used in the project.]

[Use one consistent citation format.]

1. [Author or organization. Year. Title. URL or DOI.]
2. [Dataset author or organization. Dataset title. Version or access date. URL.]
3. [Library or framework documentation. Relevant page. URL.]

---

## Appendix A. Additional Results

[Include additional tables, metrics, figures, experiments, or supporting evidence that do not fit in the main report.]

## Appendix B. AI Usage

[Document AI assistance according to the project's requirements.]

[If AI assistance beyond documentation and debugging was used, include the required AI_USAGE.md contents or an appropriate reference to that file.]

[Describe the tool used, the purpose of the assistance, the relevant parts of the work, and how AI-generated suggestions or content were verified.]

[Follow your course's AI disclosure requirements.]

## Appendix C. Reproducibility Notes

[Record any additional information needed to reproduce the experiments, such as random seeds, package versions, configuration files, data versions, or hardware requirements.]

