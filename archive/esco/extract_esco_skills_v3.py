
import pandas as pd
import re
from collections import defaultdict
from flashtext import KeywordProcessor

ESCO_PATH = "data/ESCO dataset - v1.2.1 - classification - en - csv/skills_en.csv"
JOBS_PATH = "data/processed/jobs_clean.tsv"

# Keep only well-defined short technical aliases
ALLOWED_SHORT_ALIASES = {"crm", "nmap", "webinspect"}

# Known ambiguous expressions
AMBIGUOUS = {
    "source", "sap", "clean", "scratch",
    "lead", "design", "plan", "schedule",
    "job opportunities"
}

# 1. Load ESCO dictionary
skills = pd.read_csv(
    ESCO_PATH,
    usecols=["conceptUri", "preferredLabel", "altLabels"]
).fillna("")

label_map = defaultdict(dict)

for _, row in skills.iterrows():
    uri = row["conceptUri"]
    name = row["preferredLabel"]

    labels = [name] + row["altLabels"].splitlines()

    for label in labels:
        label = " ".join(label.split()).casefold()

        if len(label) < 3 or label in AMBIGUOUS:
            continue

        # Avoid most ambiguous single-word alternative labels
        is_alternative = label != name.casefold()
        if (
            is_alternative
            and " " not in label
            and label not in ALLOWED_SHORT_ALIASES
        ):
            continue

        label_map[label][uri] = name

# 2. Build fast matching dictionary
processor = KeywordProcessor(case_sensitive=False)

for label, candidates in label_map.items():
    if len(candidates) != 1:
        continue

    uri, name = next(iter(candidates.items()))
    processor.add_keyword(label, (uri, name, label))

print("Skill patterns loaded:", len(processor))

# 3. Prepare the relevant job text
def get_relevant_text(row):
    requirements = str(row["Requirements"]).strip()
    description = str(row["Description"]).strip()

    generic_requirements = (
        len(requirements) < 50
        or "please refer to the job description" in requirements.lower()
    )

    if generic_requirements:
        source = "Description (fallback)"
        text = description
    else:
        source = "Requirements"
        text = requirements

    # Remove common benefits sections
    text = re.split(
        r"\b(?:COMPANY BENEFITS|EMPLOYEE BENEFITS|"
        r"BENEFITS|PERKS)\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    return text, source

# 4. Extract unique ESCO skills
def extract_skills(text):
    found = {}

    for uri, name, matched_label in processor.extract_keywords(text):
        if uri not in found:
            found[uri] = {
                "skill": name,
                "matched_label": matched_label,
                "esco_uri": uri
            }

    return list(found.values())

# 5. Test the three jobs we investigated
jobs = pd.read_csv(
    JOBS_PATH,
    sep="\t",
    usecols=[
        "JobID", "WindowID", "Title",
        "Description", "Requirements"
    ],
    nrows=1000,
    dtype=str,
    keep_default_na=False
)

TEST_IDS = {"454", "2723", "2389"}

for _, job in jobs.iterrows():
    if job["JobID"] not in TEST_IDS:
        continue

    text, source = get_relevant_text(job)
    matches = extract_skills(text)

    print("\n" + "=" * 55)
    print("JobID:", job["JobID"])
    print("WindowID:", job["WindowID"])
    print("Title:", job["Title"])
    print("Text source:", source)
    print("Skills found:", len(matches))

    for match in matches[:25]:
        print(
            f"- {match['skill']} "
            f"(matched: {match['matched_label']})"
        )
