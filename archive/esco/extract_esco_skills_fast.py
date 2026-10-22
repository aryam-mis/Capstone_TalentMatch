
import pandas as pd
from collections import defaultdict
from flashtext import KeywordProcessor

# File paths
ESCO_PATH = "data/ESCO dataset - v1.2.1 - classification - en - csv/skills_en.csv"
JOBS_PATH = "data/processed/jobs_clean.tsv"

# Ambiguous aliases found during testing
AMBIGUOUS_ALIASES = {
    "source",
    "sap",
    "clean",
    "job opportunities",
    "lead",
    "design"
}

# 1. Load ESCO
skills = pd.read_csv(
    ESCO_PATH,
    usecols=["conceptUri", "preferredLabel", "altLabels"]
).fillna("")

# 2. Collect preferred and alternative skill names
label_map = defaultdict(dict)

for _, row in skills.iterrows():
    uri = row["conceptUri"]
    skill_name = row["preferredLabel"]

    labels = [skill_name]
    labels.extend(row["altLabels"].splitlines())

    for label in labels:
        label = " ".join(label.split())
        key = label.casefold()

        if len(key) < 3:
            continue

        if key in AMBIGUOUS_ALIASES:
            continue

        label_map[key][uri] = skill_name

# 3. Build FlashText processor
processor = KeywordProcessor(case_sensitive=False)

ambiguous_count = 0

for label, candidates in label_map.items():

    # Skip labels shared by multiple ESCO concepts
    if len(candidates) > 1:
        ambiguous_count += 1
        continue

    uri, skill_name = next(iter(candidates.items()))

    processor.add_keyword(
        label,
        (uri, skill_name, label)
    )

print("Patterns loaded:", len(processor))
print("Ambiguous labels skipped:", ambiguous_count)

# 4. Extract unique skills
def extract_skills(text):
    matches = processor.extract_keywords(str(text))

    found = {}

    for uri, skill_name, matched_label in matches:
        if uri not in found:
            found[uri] = {
                "skill": skill_name,
                "matched_label": matched_label,
                "esco_uri": uri
            }

    return list(found.values())

# 5. Test on five jobs
jobs = pd.read_csv(
    JOBS_PATH,
    sep="\t",
    usecols=["JobID", "WindowID", "text_clean"],
    nrows=5
)

for _, job in jobs.iterrows():

    matches = extract_skills(job["text_clean"])

    print(
        f"\nJobID: {job['JobID']} | "
        f"WindowID: {job['WindowID']}"
    )

    if not matches:
        print("No skills found")

    for match in matches:
        print(
            f"- {match['skill']} "
            f"(matched: {match['matched_label']})"
        )

    print("-" * 50)
