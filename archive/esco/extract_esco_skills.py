
import pandas as pd
import re

ESCO_PATH = "data/ESCO dataset - v1.2.1 - classification - en - csv/skills_en.csv"
JOBS_PATH = "data/processed/jobs_clean.tsv"

# Load ESCO skills
skills = pd.read_csv(ESCO_PATH, usecols=["conceptUri", "preferredLabel"])

# Load a small sample of jobs
jobs = pd.read_csv(
    JOBS_PATH,
    sep="\t",
    usecols=["JobID", "WindowID", "text_clean"],
    nrows=5
)

# Extract skills using exact phrase matching
def extract_skills(text):
    text = str(text).lower()
    found = []

    for _, skill in skills.iterrows():
        label = str(skill["preferredLabel"]).strip()

        if len(label) < 3:
            continue

        pattern = r"(?<!\w)" + re.escape(label.lower()) + r"(?!\w)"

        if re.search(pattern, text):
            found.append({
                "skill": label,
                "esco_uri": skill["conceptUri"]
            })

    return found

for _, job in jobs.iterrows():
    matches = extract_skills(job["text_clean"])

    print(f"\nJobID: {job['JobID']} | WindowID: {job['WindowID']}")
    print("Extracted skills:", [m["skill"] for m in matches])
