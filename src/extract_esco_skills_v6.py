
"""TalentMatch: ESCO skills extraction V6.

Test on 100 jobs:
    python src/extract_esco_skills_v6.py

Full extraction after evaluation:
    python src/extract_esco_skills_v6.py --full --windows 1,2

Coverage is not precision or recall.
Human validation is required.
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

JOBS_FILE = Path(
    "data/processed/jobs_clean.tsv"
)

DEFAULT_SAMPLE = Path(
    "data/processed/esco_holdout_100.csv"
)

DEFAULT_SAMPLE_OUT = Path(
    "data/processed/esco_holdout_100_v6_predictions.csv"
)

DEFAULT_FULL_OUT = Path(
    "data/processed/jobs_esco_skills_v6.tsv"
)

V5_SAMPLE = Path(
    "data/processed/esco_holdout_100_v5_predictions.csv"
)

COMPARE_OUT = Path(
    "data/processed/esco_v5_vs_v6_review.csv"
)

INPUT_COLUMNS = [
    "JobID",
    "WindowID",
    "Title",
    "Requirements",
    "Description",
]

PREDICTION_COLUMNS = [
    "JobID",
    "WindowID",
    "Title",
    "TextSource",
    "ExtractedSkills",
    "ESCO_URIs",
    "MatchedLabels",
    "TextForReview",
]


# ==========================================
# 2. FILTER AMBIGUOUS LABELS
# ==========================================

AMBIGUOUS_LABELS = {
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

BAD_ALTERNATIVE_LABELS = {
    "visual studio",
    "search engine marketing",
    "marketing techniques",
    "medical equipment",
    "state regulations",
    "food production",
    "creative solutions",
}

ALLOWED_SHORT_ALIASES = {
    "crm",
    "nmap",
    "webinspect",
    "cpr",
}


# ==========================================
# 3. CONTEXT RULES
# ==========================================

CONTEXT_RULES = {
    "hair": re.compile(
        r"\b(?:hairdress\w*|hairstyl\w*|"
        r"hair styl\w*|barber\w*|cosmetolog\w*|"
        r"hair salon|hair care)\b",
        re.I,
    ),

    "history": re.compile(
        r"\b(?:historian\w*|history teacher|"
        r"history professor|historical research|"
        r"historical archives|historical studies)\b",
        re.I,
    ),

    "energy": re.compile(
        r"\b(?:energy efficiency|energy consumption|"
        r"renewable energy|energy management|"
        r"energy systems|energy production|"
        r"energy engineering|energy sector|"
        r"energy analyst)\b",
        re.I,
    ),

    "logic": re.compile(
        r"\b(?:logical reasoning|formal logic|"
        r"logic programming|logic design|"
        r"logical circuits)\b",
        re.I,
    ),
}


ROOT_CAUSE = re.compile(
    r"\b(?:identif(?:y|ies|ied|ying)|"
    r"investigat(?:e|es|ed|ing)|"
    r"determin(?:e|es|ed|ing)|"
    r"analy[sz](?:e|es|ed|ing))"
    r"\b.{0,80}\broot causes?\b",
    re.I,
)


# ==========================================
# 4. FILTER NON-JOB CONTENT
# ==========================================

FOOTER_MARKERS = (
    re.compile(
        r"\b(?:company\s+benefits|employee\s+benefits|"
        r"benefits|perks)\s*"
        r"(?::|at\s+[A-Z]|include\b|including\b|"
        r"we\s+offer\b)",
        re.I,
    ),

    re.compile(
        r"\b(?:an?\s+)?"
        r"(?:affirmative\s+action\s*[/,&-]\s*)?"
        r"(?:equal\s+opportunity\s+employers?|"
        r"equal\s+employment\s+opportunity)\b",
        re.I,
    ),

    re.compile(
        r"\ball\s+qualified\s+applicants\s+"
        r"will\s+receive\s+consideration\b",
        re.I,
    ),

    re.compile(
        r"\bwithout\s+regard\s+to\s+"
        r"(?:race|color|religion|sex|gender)\b",
        re.I,
    ),

    re.compile(
        r"\b(?:about\s+us|about\s+the\s+company|"
        r"company\s+overview)\s*:?\b",
        re.I,
    ),

    re.compile(
        r"\b(?:search\s+terms|keywords)\s*:",
        re.I,
    ),
)


# ==========================================
# 5. IDENTIFY JOB SECTIONS
# ==========================================

ROLE_HEADING = re.compile(
    r"\b(?:"
    r"essential\s+(?:job\s+)?duties"
    r"(?:\s+and\s+responsibilities)?"
    r"(?:\s+include\s+but\s+are\s+not\s+limited\s+to)?|"

    r"(?:key|main|primary|job|general|specific)"
    r"\s+responsibilities|"

    r"responsibilities|"

    r"duties(?:\s+may\s+include)?"
    r"(?:\s+but\s+are\s+not\s+limited\s+to"
    r"(?:\s+the\s+following)?)?|"

    r"(?:selected\s+candidates\s+will\s+be\s+)?"
    r"responsible\s+for\s+the\s+following|"

    r"(?:minimum|required|preferred|job)"
    r"\s+requirements|"

    r"requirements(?:\s+for\s+[^:]{3,65})?|"

    r"(?:required|desired|minimum|job)"
    r"\s+qualifications|"

    r"qualifications|"

    r"(?:essential|key)\s+functions|"

    r"(?:knowledge|skills)\s*(?:/|and)"
    r"\s*(?:skills|abilities|required)|"

    r"tasks|"

    r"job\s+description|"

    r"department\s+description\s+of\s+[^:]{2,40}"

    r")\s*:",
    re.I,
)

GENERIC_REQUIREMENTS = re.compile(
    r"please\s+refer\s+to\s+the\s*job\s*description|"
    r"jobfair|jobboardnetwork",
    re.I,
)


# ==========================================
# 6. LOAD ESCO TAXONOMY
# ==========================================

def build_matcher():

    if not ESCO_FILE.is_file():
        raise FileNotFoundError(
            f"Missing ESCO taxonomy: {ESCO_FILE}"
        )

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

        name = " ".join(
            row.preferredLabel.split()
        )

        preferred = name.casefold()

        labels = [
            name,
            *row.altLabels.splitlines(),
        ]

        for raw in labels:

            label = " ".join(
                raw.split()
            ).casefold()

            if (
                len(label) < 3
                or label in AMBIGUOUS_LABELS
            ):
                continue

            alternative = label != preferred

            if (
                alternative
                and label in BAD_ALTERNATIVE_LABELS
            ):
                continue

            if (
                alternative
                and " " not in label
                and label not in ALLOWED_SHORT_ALIASES
            ):
                continue

            label_map[label][
                row.conceptUri
            ] = name

    matcher = KeywordProcessor(
        case_sensitive=False
    )

    for label, concepts in label_map.items():

        if len(concepts) != 1:
            continue

        uri, name = next(
            iter(concepts.items())
        )

        matcher.add_keyword(
            label,
            (uri, name, label),
        )

    root_rows = skills[
        skills["preferredLabel"].str.casefold()
        == "root cause analysis"
    ]

    root_uri = (
        root_rows.iloc[0]["conceptUri"]
        if len(root_rows) == 1
        else None
    )

    print(
        f"V6 READY | ESCO patterns loaded: "
        f"{len(matcher):,}"
    )

    return matcher, root_uri


# ==========================================
# 7. REMOVE BOILERPLATE
# ==========================================

def trim_boilerplate(text):

    text = " ".join(
        str(text or "").split()
    )

    positions = []

    for marker in FOOTER_MARKERS:

        match = marker.search(text)

        if match:
            positions.append(
                match.start()
            )

    if positions:
        text = text[
            :min(positions)
        ].strip()

    return text


# ==========================================
# 8. SELECT RELEVANT JOB TEXT
# ==========================================

def role_text(description):

    description = " ".join(
        str(description or "").split()
    )

    heading = ROLE_HEADING.search(
        description
    )

    duty_lead = re.search(
        r"\b(?:essential\s+)?"
        r"job\s+duties\s+include\b",
        description,
        re.I,
    )

    if (
        duty_lead
        and (
            heading is None
            or duty_lead.start() < heading.start()
        )
    ):
        heading = duty_lead

    if heading:

        relevant = description[
            heading.start():
        ]

        return (
            trim_boilerplate(relevant),
            True,
        )

    return (
        trim_boilerplate(description),
        False,
    )


# ==========================================
# 9. EXTRACT SKILLS
# ==========================================

def extract_skills(
    text,
    title,
    matcher,
    root_uri,
):

    text = str(text or "")

    context = f"{title}\n{text}"

    found = {}

    matches = matcher.extract_keywords(
        text
    )

    for uri, name, label in matches:

        rule = CONTEXT_RULES.get(
            label
        )

        if (
            rule is not None
            and not rule.search(context)
        ):
            continue

        found[uri] = {
            "skill": name,
            "matched_label": label,
            "esco_uri": uri,
        }

    if (
        root_uri
        and ROOT_CAUSE.search(text)
    ):

        found[root_uri] = {
            "skill": "root cause analysis",
            "matched_label": "root cause (context rule)",
            "esco_uri": root_uri,
        }

    return list(found.values())


# ==========================================
# 10. PROCESS ONE JOB
# ==========================================

def predict(row, matcher, root_uri):

    title = str(
        row.get("Title", "")
    )

    req = trim_boilerplate(
        row.get("Requirements", "")
    )

    desc, desc_sectioned = role_text(
        row.get("Description", "")
    )

    req_is_generic = (
        len(req) < 120
        or bool(
            GENERIC_REQUIREMENTS.search(req)
        )
    )

    if desc_sectioned and desc:

        if req and not req_is_generic:
            pieces = [req, desc]
        else:
            pieces = [desc]

        text = "\n".join(pieces)

        source = (
            "Requirements + role description"
            if len(pieces) == 2
            else "Description (role sections)"
        )

        matches = extract_skills(
            text,
            title,
            matcher,
            root_uri,
        )

    else:

        req_matches = extract_skills(
            req,
            title,
            matcher,
            root_uri,
        )

        if (
            req_matches
            and not req_is_generic
        ):

            text = req
            source = "Requirements"
            matches = req_matches

        elif desc:

            desc_matches = extract_skills(
                desc,
                title,
                matcher,
                root_uri,
            )

            if (
                desc_matches
                or not req_matches
            ):

                text = desc
                source = "Description (safe fallback)"
                matches = desc_matches

            else:

                text = req
                source = "Requirements"
                matches = req_matches

        else:

            text = req
            source = "Requirements"
            matches = req_matches

    return {
        "JobID": str(
            row.get("JobID", "")
        ),

        "WindowID": str(
            row.get("WindowID", "")
        ),

        "Title": title,

        "TextSource": source,

        "ExtractedSkills": "; ".join(
            item["skill"]
            for item in matches
        ),

        "ESCO_URIs": "; ".join(
            item["esco_uri"]
            for item in matches
        ),

        "MatchedLabels": "; ".join(
            item["matched_label"]
            for item in matches
        ),

        "TextForReview": text,
    }


# ==========================================
# 11. COMPARE WITH V5
# ==========================================

def compare_v5(
    rows,
    old_path,
    output_path,
):

    if not old_path.exists():

        print(
            f"V5 comparison skipped: {old_path}"
        )

        return

    old = pd.read_csv(
        old_path,
        dtype=str,
        keep_default_na=False,
    )

    previous = {
        (r.JobID, r.WindowID): r
        for r in old.itertuples(index=False)
    }

    changes = []

    for row in rows:

        key = (
            row["JobID"],
            row["WindowID"],
        )

        if key not in previous:
            continue

        previous_row = previous[key]

        old_skills = {
            skill.strip()
            for skill in previous_row.ExtractedSkills.split(";")
            if skill.strip()
        }

        new_skills = {
            skill.strip()
            for skill in row["ExtractedSkills"].split(";")
            if skill.strip()
        }

        if old_skills == new_skills:
            continue

        changes.append({
            "JobID": row["JobID"],
            "WindowID": row["WindowID"],
            "Title": row["Title"],

            "V5": previous_row.ExtractedSkills,
            "V6": row["ExtractedSkills"],

            "Removed": "; ".join(
                sorted(old_skills - new_skills)
            ),

            "Added": "; ".join(
                sorted(new_skills - old_skills)
            ),
        })

    columns = [
        "JobID",
        "WindowID",
        "Title",
        "V5",
        "V6",
        "Removed",
        "Added",
    ]

    pd.DataFrame(
        changes,
        columns=columns,
    ).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Jobs changed versus V5: "
        f"{len(changes)} / {len(rows)}"
    )

    print(
        f"Changes saved: {output_path}"
    )

    print(
        "Coverage and changes are not "
        "precision or recall."
    )


# ==========================================
# 12. TEST ON 100 JOBS
# ==========================================

def run_sample(
    args,
    matcher,
    root_uri,
):

    if not args.sample_path.exists():

        raise FileNotFoundError(
            f"Missing sample: {args.sample_path}"
        )

    jobs = pd.read_csv(
        args.sample_path,
        usecols=INPUT_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )

    rows = [
        predict(row, matcher, root_uri)
        for row in jobs.to_dict("records")
    ]

    target = (
        args.output
        or DEFAULT_SAMPLE_OUT
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows,
        columns=PREDICTION_COLUMNS,
    ).to_csv(
        target,
        index=False,
        encoding="utf-8-sig",
    )

    with_skills = sum(
        bool(row["ExtractedSkills"])
        for row in rows
    )

    print(
        f"Sample tested: {len(rows)} jobs"
    )

    print(
        f"Jobs with >=1 extracted skill: "
        f"{with_skills}"
    )

    print(
        f"Predictions saved: {target}"
    )

    if (
        args.sample_path.resolve()
        == DEFAULT_SAMPLE.resolve()
    ):

        compare_v5(
            rows,
            V5_SAMPLE,
            COMPARE_OUT,
        )

    else:

        print(
            "V5 comparison skipped "
            "for a different sample."
        )


# ==========================================
# 13. FULL DATASET PROCESSING
# ==========================================

def run_full(
    args,
    matcher,
    root_uri,
):

    if not JOBS_FILE.exists():

        raise FileNotFoundError(
            f"Missing cleaned jobs: {JOBS_FILE}"
        )

    target = (
        args.output
        or DEFAULT_FULL_OUT
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_ids = (
        set(args.windows.split(","))
        if args.windows
        else None
    )

    output_columns = [
        col
        for col in PREDICTION_COLUMNS
        if col != "TextForReview"
    ]

    processed = 0

    with target.open(
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
            usecols=INPUT_COLUMNS,
            dtype=str,
            keep_default_na=False,
            chunksize=1000,
        )

        for chunk in chunks:

            if window_ids is not None:

                chunk = chunk[
                    chunk["WindowID"].isin(
                        window_ids
                    )
                ]

            for row in chunk.to_dict("records"):

                result = predict(
                    row,
                    matcher,
                    root_uri,
                )

                result.pop(
                    "TextForReview"
                )

                writer.writerow(result)

                processed += 1

            if (
                processed
                and processed % 50000 == 0
            ):

                print(
                    f"Processed {processed:,} jobs",
                    flush=True,
                )

    print(
        f"Finished {processed:,} jobs "
        f"-> {target}"
    )


# ==========================================
# 14. MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "TalentMatch ESCO Skills V6"
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Process cleaned jobs dataset",
    )

    parser.add_argument(
        "--windows",
        help="WindowIDs, e.g. 1,2",
    )

    parser.add_argument(
        "--sample-path",
        type=Path,
        default=DEFAULT_SAMPLE,
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    args = parser.parse_args()

    matcher, root_uri = build_matcher()

    if args.full:

        run_full(
            args,
            matcher,
            root_uri,
        )

    else:

        run_sample(
            args,
            matcher,
            root_uri,
        )


if __name__ == "__main__":
    main()
