
import re
from collections import defaultdict

import pandas as pd
from flashtext import KeywordProcessor

ESCO_PATH = "data/ESCO dataset - v1.2.1 - classification - en - csv/skills_en.csv"
JOBS_PATH = "data/processed/jobs_clean.tsv"

AMBIGUOUS = {
    "source", "sap", "clean", "scratch",
    "lead", "design", "plan", "schedule",
    "job opportunities"
}

ALLOWED_SHORT_ALIASES = {"crm", "nmap", "webinspect"}

# 1. Load ESCO
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

        is_alternative = label != name.casefold()

        if (
            is_alternative
            and " " not in label
            and label not in ALLOWED_SHORT_ALIASES
        ):
            continue

        label_map[label][uri] = name

# 2. Build FlashText matcher
processor = KeywordProcessor(case_sensitive=False)

for label, candidates in label_map.items():
    if len(candidates) != 1:
        continue

    uri, name = next(iter(candidates.items()))
    processor.add_keyword(label, (uri, name, label))

print("Skill patterns loaded:", len(processor))

# 3. Find root cause analysis in ESCO
root_rows = skills[
    skills["preferredLabel"].str.casefold() == "root cause analysis"
]

ROOT_CAUSE_URI = (
    root_rows.iloc[0]["conceptUri"]
    if not root_rows.empty else None
)

ROOT_CAUSE_PATTERN = re.compile(
    r"\b(?:identif(?:y|ies|ied|ying)|"
    r"investigat(?:e|es|ed|ing)|"
    r"determin(?:e|es|ed|ing)|"
    r"analy[sz](?:e|es|ed|ing))"
    r"\b.{0,80}\broot causes?\b",
    re.IGNORECASE
)

# 4. Remove common benefits sections
def clean_section(text):
    text = str(text).strip()

    return re.split(
        r"\b(?:COMPANY BENEFITS|EMPLOYEE BENEFITS|"
        r"BENEFITS|PERKS)\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]


# 5. Extract skills
def extract_skills(text):
    text = str(text)
    found = {}

    for uri, name, matched_label in processor.extract_keywords(text):
        found[uri] = {
            "skill": name,
            "matched_label": matched_label,
            "esco_uri": uri
        }

    # Experimental contextual rule
    if ROOT_CAUSE_URI and ROOT_CAUSE_PATTERN.search(text):
        found[ROOT_CAUSE_URI] = {
            "skill": "root cause analysis",
            "matched_label": "root cause (context rule)",
            "esco_uri": ROOT_CAUSE_URI
        }

    return list(found.values())


# 6. Select relevant job text
def get_relevant_text(row):
    req = clean_section(row["Requirements"])
    desc = clean_section(row["Description"])

    req_lower = req.lower()

    generic_req = (
        len(req) < 120
        or bool(re.search(
            r"please\s+refer\s+to\s+the\s*job\s*description",
            req_lower
        ))
        or "jobfair" in req_lower
        or "jobboardnetwork" in req_lower
    )

    if generic_req and desc:
        return desc, "Description (fallback)"

    if not extract_skills(req) and desc:
        return desc, "Description (no matches in Requirements)"

    return req, "Requirements"


# 7. Read a small test sample
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

TEST_IDS = {"454", "2389", "2723", "3253", "2453", "1843"}

for _, job in jobs.iterrows():
    if job["JobID"] not in TEST_IDS:
        continue

    text, source = get_relevant_text(job)
    matches = extract_skills(text)

    print("\n" + "=" * 55)
    print("JobID:", job["JobID"])
    print("WindowID:", job["WindowID"])
    print("Title:", job["Title"])
    print("Source:", source)
    print("Skills found:", len(matches))

    for match in matches[:20]:
        print(
            f"- {match['skill']} "
            f"(matched: {match['matched_label']})"
        )
