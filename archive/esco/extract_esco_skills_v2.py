
import pandas as pd
import re

# File paths
ESCO_PATH = "data/ESCO dataset - v1.2.1 - classification - en - csv/skills_en.csv"
JOBS_PATH = "data/processed/jobs_clean.tsv"

# Ambiguous aliases that cause false positives
AMBIGUOUS_ALIASES = {
    "source",
    "sap",
    "clean"
}

# 1. Load ESCO skills
skills = pd.read_csv(
    ESCO_PATH,
    usecols=["conceptUri", "preferredLabel", "altLabels"]
)

# 2. Build skill patterns
skill_patterns = []

for _, row in skills.iterrows():

    labels = [row["preferredLabel"]]

    if pd.notna(row["altLabels"]):
        labels.extend(str(row["altLabels"]).splitlines())

    for label in labels:

        if pd.isna(label):
            continue

        label = str(label).strip()

        # Ignore very short labels
        if len(label) < 3:
            continue

        # Ignore ambiguous aliases
        if label.casefold() in AMBIGUOUS_ALIASES:
            continue

        # Match complete words or phrases
        pattern = re.compile(
            r"(?<!\w)" + re.escape(label) + r"(?!\w)",
            re.IGNORECASE
        )

        skill_patterns.append(
            (
                pattern,
                row["preferredLabel"],
                row["conceptUri"]
            )
        )

print("Skill patterns loaded:", len(skill_patterns))

# 3. Load five jobs for testing
jobs = pd.read_csv(
    JOBS_PATH,
    sep="\t",
    usecols=["JobID", "WindowID", "text_clean"],
    nrows=5
)

# 4. Extract skills from job descriptions
def extract_skills(text):

    found = {}
    text = str(text)

    for pattern, skill_name, uri in skill_patterns:

        if pattern.search(text):
            found[uri] = skill_name

    return list(found.values())


# 5. Display extracted skills
for _, job in jobs.iterrows():

    matches = extract_skills(job["text_clean"])

    print(
        f"\nJobID: {job['JobID']} | "
        f"WindowID: {job['WindowID']}"
    )

    print("Extracted skills:", matches)
    print("-" * 50)
