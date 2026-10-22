
"""TalentMatch ESCO Skills Extraction V5.

Test on 50 jobs:
    python src/extract_esco_skills_v5.py

Full dataset (after evaluation):
    python src/extract_esco_skills_v5.py --full
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from flashtext import KeywordProcessor


# ==========================================
# 1. FILE PATHS
# ==========================================

ESCO_FILE = Path(
    "data/ESCO dataset - v1.2.1 - classification - en - csv/skills_en.csv"
)
SAMPLE_FILE = Path("data/processed/esco_review_sample.csv")
JOBS_FILE = Path("data/processed/jobs_clean.tsv")
V4_FILE = Path("data/processed/esco_review_predictions.csv")
V5_FILE = Path("data/processed/esco_v5_review_predictions.csv")
COMPARE_FILE = Path("data/processed/esco_v4_vs_v5.csv")
FULL_FILE = Path("data/processed/jobs_esco_skills_v5.tsv")

INPUT_COLUMNS = [
    "JobID",
    "WindowID",
    "Title",
    "Requirements",
    "Description",
]


# ==========================================
# 2. FILTER AMBIGUOUS SKILL LABELS
# ==========================================

BLACKLIST = {
    "source",
    "sap",
    "clean",
    "scratch",
    "lead",
    "design",
    "plan",
    "schedule",
    "job opportunities",
}

BAD_ALTERNATIVES = {
    "visual studio",
    "search engine marketing",
    "marketing techniques",
    "medical equipment",
    "state regulations",
    "food production",
}

ALLOWED_SHORT_ALIASES = {
    "crm",
    "nmap",
    "webinspect",
}


# ==========================================
# 3. CONTEXT RULES
# ==========================================

CONTEXT_RULES = {
    "hair": (
        r"\b(?:hairdress\w*|hairstyl\w*|hair styl\w*|"
        r"barber\w*|cosmetolog\w*|hair salon|hair care)\b"
    ),
    "history": (
        r"\b(?:historian\w*|history teacher|history professor|"
        r"historical research|historical archives|historical studies)\b"
    ),
    "energy": (
        r"\b(?:energy efficiency|energy consumption|renewable energy|"
        r"energy management|energy systems|energy production|"
        r"energy engineering|energy sector|energy analyst)\b"
    ),
    "logic": (
        r"\b(?:logical reasoning|formal logic|logic programming|"
        r"logic design|logical circuits)\b"
    ),
}

CONTEXT_RULES = {
    label: re.compile(pattern, re.IGNORECASE)
    for label, pattern in CONTEXT_RULES.items()
}

ROOT_CAUSE_RE = re.compile(
    r"\b(?:identif(?:y|ies|ied|ying)|"
    r"investigat(?:e|es|ed|ing)|"
    r"determin(?:e|es|ed|ing)|"
    r"analy[sz](?:e|es|ed|ing))"
    r"\b.{0,80}\broot causes?\b",
    re.IGNORECASE,
)

BENEFITS_RE = re.compile(
    r"\b(?:company benefits|employee benefits|"
    r"benefits|perks)\s*:",
    re.IGNORECASE,
)


# ==========================================
# 4. BUILD ESCO MATCHER
# ==========================================

def build_matcher():

    skills = pd.read_csv(
        ESCO_FILE,
        usecols=[
            "conceptUri",
            "preferredLabel",
            "altLabels",
        ],
        dtype=str,
        keep_default_na=False,
    )

    label_map = defaultdict(dict)

    for row in skills.itertuples(index=False):

        name = row.preferredLabel
        preferred = " ".join(name.split()).casefold()

        labels = [
            name,
            *row.altLabels.splitlines(),
        ]

        for raw_label in labels:

            label = " ".join(raw_label.split()).casefold()
            is_alternative = label != preferred

            if len(label) < 3 or label in BLACKLIST:
                continue

            if is_alternative and label in BAD_ALTERNATIVES:
                continue

            if (
                is_alternative
                and " " not in label
                and label not in ALLOWED_SHORT_ALIASES
            ):
                continue

            label_map[label][row.conceptUri] = name

    matcher = KeywordProcessor(case_sensitive=False)

    for label, candidates in label_map.items():

        if len(candidates) == 1:

            uri, name = next(iter(candidates.items()))

            matcher.add_keyword(
                label,
                (uri, name, label),
            )

    roots = skills[
        skills["preferredLabel"].str.casefold()
        == "root cause analysis"
    ]

    root_uri = (
        roots.iloc[0]["conceptUri"]
        if len(roots) == 1
        else None
    )

    print(
        f"V5 READY | ESCO patterns loaded: {len(matcher):,}"
    )

    return matcher, root_uri


# ==========================================
# 5. CLEAN JOB TEXT
# ==========================================

def clean_text(value):

    text = str(value).strip()

    return BENEFITS_RE.split(
        text,
        maxsplit=1,
    )[0]


# ==========================================
# 6. EXTRACT ESCO SKILLS
# ==========================================

def extract_skills(text, title, matcher, root_uri):

    found = {}
    context = f"{title}\n{text}"

    matches = matcher.extract_keywords(text)

    for uri, skill, label in matches:

        rule = CONTEXT_RULES.get(label)

        if rule is not None and not rule.search(context):
            continue

        found[uri] = {
            "skill": skill,
            "matched_label": label,
            "esco_uri": uri,
        }

    if root_uri and ROOT_CAUSE_RE.search(text):

        found[root_uri] = {
            "skill": "root cause analysis",
            "matched_label": "root cause (context rule)",
            "esco_uri": root_uri,
        }

    return list(found.values())


# ==========================================
# 7. PREDICT SKILLS FOR ONE JOB
# ==========================================

def predict(row, matcher, root_uri):

    title = str(row.get("Title", ""))

    requirements = clean_text(
        row.get("Requirements", "")
    )

    description = clean_text(
        row.get("Description", "")
    )

    generic_requirements = (
        len(requirements) < 120
        or bool(
            re.search(
                r"please\s+refer\s+to\s+the\s*job\s*description",
                requirements,
                re.IGNORECASE,
            )
        )
        or "jobfair" in requirements.casefold()
        or "jobboardnetwork" in requirements.casefold()
    )

    if generic_requirements and description:

        text = description
        source = "Description (fallback)"

        section = re.search(
            r"\b(?:essential\s+duties(?:\s+and\s+responsibilities)?\b|"
            r"duties\s+may\s+include\b|"
            r"responsibilities\s*:|responsible\s+for\s+the\s+following\s*:|"
            r"(?:minimum\s+|required\s+)?requirements\s*:|"
            r"(?:required\s+|desired\s+)?qualifications\s*:)",
            description,
            re.IGNORECASE,
        )

        if section:
            text = description[section.start():].strip()
            source = "Description (duties + requirements)"

        matches = extract_skills(
            text,
            title,
            matcher,
            root_uri,
        )

    else:

        text = requirements
        source = "Requirements"

        matches = extract_skills(
            text,
            title,
            matcher,
            root_uri,
        )

        if not matches and description:

            text = description
            source = "Description (no matches in Requirements)"

            matches = extract_skills(
                text,
                title,
                matcher,
                root_uri,
            )

    return {
        "JobID": str(row.get("JobID", "")),
        "WindowID": str(row.get("WindowID", "")),
        "Title": title,
        "TextSource": source,

        "ExtractedSkills": "; ".join(
            m["skill"] for m in matches
        ),

        "ESCO_URIs": "; ".join(
            m["esco_uri"] for m in matches
        ),

        "MatchedLabels": "; ".join(
            m["matched_label"] for m in matches
        ),

        "TextForReview": text,
    }


# ==========================================
# 8. COMPARE V4 AND V5
# ==========================================

def compare_with_v4(v5_rows):

    if not V4_FILE.exists():
        print("V4 predictions not found. Comparison skipped.")
        return

    v4 = pd.read_csv(
        V4_FILE,
        dtype=str,
        keep_default_na=False,
    )

    old_by_id = dict(
        zip(
            v4["JobID"],
            v4["ExtractedSkills"],
        )
    )

    comparisons = []

    for row in v5_rows:

        job_id = row["JobID"]

        if job_id not in old_by_id:
            continue

        before = {
            s.strip()
            for s in old_by_id[job_id].split(";")
            if s.strip()
        }

        after = {
            s.strip()
            for s in row["ExtractedSkills"].split(";")
            if s.strip()
        }

        comparisons.append({
            "JobID": job_id,
            "Title": row["Title"],
            "V4_count": len(before),
            "V5_count": len(after),

            "Removed_from_V4": "; ".join(
                sorted(before - after)
            ),

            "Added_in_V5": "; ".join(
                sorted(after - before)
            ),

            "NeedsManualCheck": (
                "Yes" if before != after else "No"
            ),
        })

    pd.DataFrame(comparisons).to_csv(
        COMPARE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    changed = sum(
        row["NeedsManualCheck"] == "Yes"
        for row in comparisons
    )

    print(
        f"V4 vs V5: {changed} changed jobs "
        f"out of {len(comparisons)}"
    )

    print(f"Comparison saved: {COMPARE_FILE}")

    print(
        "Note: Changes do not automatically mean better accuracy."
    )


# ==========================================
# 9. TEST ON 50 JOBS
# ==========================================

def run_sample(matcher, root_uri):

    sample = pd.read_csv(
        SAMPLE_FILE,
        dtype=str,
        keep_default_na=False,
        usecols=INPUT_COLUMNS,
    )

    predictions = []

    for row in sample.to_dict("records"):

        result = predict(
            row,
            matcher,
            root_uri,
        )

        predictions.append(result)

    pd.DataFrame(predictions).to_csv(
        V5_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    with_skills = sum(
        bool(row["ExtractedSkills"])
        for row in predictions
    )

    print(f"Sample tested: {len(predictions)} jobs")
    print(f"Jobs with skills: {with_skills}")
    print(f"Predictions saved: {V5_FILE}")

    compare_with_v4(predictions)


# ==========================================
# 10. PROCESS FULL DATASET
# ==========================================

def run_full(matcher, root_uri):

    FULL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "JobID",
        "WindowID",
        "Title",
        "TextSource",
        "ExtractedSkills",
        "ESCO_URIs",
        "MatchedLabels",
    ]

    processed = 0

    with FULL_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=output_columns,
            delimiter="\t",
        )

        writer.writeheader()

        chunks = pd.read_csv(
            JOBS_FILE,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=INPUT_COLUMNS,
            chunksize=1000,
        )

        for chunk in chunks:

            for row in chunk.to_dict("records"):

                result = predict(
                    row,
                    matcher,
                    root_uri,
                )

                result.pop("TextForReview")

                writer.writerow(result)

                processed += 1

            if processed % 50000 == 0:

                print(
                    f"Processed {processed:,} jobs",
                    flush=True,
                )

    print(
        f"Finished {processed:,} jobs -> {FULL_FILE}"
    )


# ==========================================
# 11. MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="ESCO Skills Extraction V5"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Process the entire jobs dataset",
    )

    args = parser.parse_args()

    matcher, root_uri = build_matcher()

    if args.full:
        run_full(matcher, root_uri)
    else:
        run_sample(matcher, root_uri)


if __name__ == "__main__":
    main()
