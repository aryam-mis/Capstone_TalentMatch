
"""TalentMatch - ESCO Skills Extraction.

Full extraction for Window 1:
    python src/extract_esco_skills_candidate.py --full --windows 1 --output data/processed/jobs_esco_window1_final.tsv

Sample test:
    python src/extract_esco_skills_candidate.py

Fast self-test:
    python src/extract_esco_skills_candidate.py --self-test

Coverage is NOT precision or recall.
Independent human evaluation is still required.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from flashtext import KeywordProcessor


# ==================================================
# 1. FILE PATHS
# ==================================================

ESCO_FILE = Path(
    "data/ESCO dataset - v1.2.1 - classification - en - csv/skills_en.csv"
)

JOBS_FILE = Path(
    "data/processed/jobs_clean.tsv"
)

DEFAULT_SAMPLE = Path(
    "data/processed/esco_holdout_100.csv"
)

SAMPLE_OUTPUT = Path(
    "data/processed/esco_candidate_100_predictions.csv"
)

FULL_OUTPUT = Path(
    "data/processed/jobs_esco_candidate.tsv"
)

V6_PREDICTIONS = Path(
    "data/processed/esco_holdout_100_v6_predictions.csv"
)

COMPARE_OUTPUT = Path(
    "data/processed/esco_candidate_vs_v6.csv"
)

INPUT_COLUMNS = [
    "JobID",
    "WindowID",
    "Title",
    "Requirements",
    "Description",
]

OUTPUT_COLUMNS = [
    "JobID",
    "WindowID",
    "Title",
    "TextSource",
    "ExtractedSkills",
    "ESCO_URIs",
    "MatchedLabels",
    "ReviewFlag",
    "TextForReview",
]


# ==================================================
# 2. FILTER AMBIGUOUS SKILLS
# ==================================================

BLOCKED_LABELS = {
    "source",
    "sap",
    "clean",
    "scratch",
    "lead",
    "design",
    "plan",
    "schedule",
    "job opportunities",
    "less",
}

BLOCKED_ALTERNATIVES = {
    "visual studio",
    "search engine marketing",
    "search engine",
    "marketing techniques",
    "medical equipment",
    "state regulations",
    "food production",
    "creative solutions",
    "operational excellence",
    "environmental laws",
    "military training",
    "be responsible",
    "managing project",
}

ALLOWED_ACRONYM_ALIASES = {
    "crm",
    "nmap",
    "webinspect",
    "cpr",
}


# ==================================================
# 3. CONTEXT RULES
# ==================================================

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

ROOT_CAUSE_RE = re.compile(
    r"\b(?:identif(?:y|ies|ied|ying)|"
    r"investigat(?:e|es|ed|ing)|"
    r"determin(?:e|es|ed|ing)|"
    r"analy[sz](?:e|es|ed|ing))"
    r"\b.{0,80}\broot causes?\b",
    re.I,
)


# ==================================================
# 4. FILTER NON-JOB CONTENT
# ==================================================

FOOTER_RE = re.compile(
    r"\b(?:"
    r"(?:an?\s+)?"
    r"(?:affirmative\s+action\s*[/,&-]\s*)?"
    r"(?:equal\s+opportunity\s+employers?|"
    r"equal\s+employment\s+opportunity)|"
    r"all\s+qualified\s+applicants\s+"
    r"will\s+receive\s+consideration|"
    r"without\s+regard\s+to\s+"
    r"(?:race|color|religion|sex|gender)|"
    r"(?:company\s+benefits|employee\s+benefits|"
    r"benefits|perks)\s*"
    r"(?::|at\s+[A-Z]|include\b|including\b|"
    r"we\s+offer\b)|"
    r"about\s+us\b|"
    r"about\s+the\s+company\b|"
    r"company\s+overview\s*:|"
    r"(?:search\s+terms|keywords)\s*:"
    r")",
    re.I,
)

GENERIC_REQUIREMENTS = re.compile(
    r"please\s+refer\s+to\s+the\s*job\s*description|"
    r"jobfair|jobboardnetwork",
    re.I,
)


# ==================================================
# 5. IDENTIFY JOB SECTIONS
# ==================================================

ROLE_HEADING = re.compile(
    r"\b(?:"
    r"(?:job\s+)?summary|"
    r"job\s+description|"
    r"description\s+of\s+responsibilities|"
    r"(?:essential|key|main|primary|job|general|specific)"
    r"\s+(?:job\s+)?duties"
    r"(?:\s+and\s+responsibilities)?|"
    r"duties(?:\s+may\s+include)?"
    r"(?:\s+but\s+are\s+not\s+limited\s+to"
    r"(?:\s+the\s+following)?)?|"
    r"(?:key|main|primary|job|general|specific)"
    r"\s+responsibilities|"
    r"responsibilities(?:\s+of\s+[^:]{3,70})?|"
    r"description\s+of\s+environment|"
    r"(?:selected\s+candidates\s+will\s+be\s+)?"
    r"responsible\s+for\s+the\s+following|"
    r"(?:minimum|required|preferred|job)"
    r"\s+requirements|"
    r"requirements(?:\s+for\s+[^:]{3,70})?|"
    r"(?:required|desired|minimum|job)"
    r"\s+qualifications|"
    r"qualifications|"
    r"(?:essential|key)\s+functions|"
    r"(?:knowledge|skills)\s*(?:/|and)"
    r"\s*(?:skills|abilities|required)|"
    r"department\s+description\s+of\s+[^:]{2,40}|"
    r"tasks"
    r")\s*:",
    re.I,
)

DUTIES_LEAD = re.compile(
    r"\b(?:essential\s+)?job\s+duties\s+include\b|"
    r"\bresponsible\s+for\s+the\s+following\s*:",
    re.I,
)

LATER_DUTIES = re.compile(
    r"\b(?:"
    r"this\s+position\s+is\s+responsible(?:\s+for)?|"
    r"(?:essential|key|primary|general)\s+"
    r"(?:duties(?:\s+and\s+responsibilities)?|"
    r"responsibilities)\b|"
    r"(?:job\s+)?"
    r"(?:duties|responsibilities|requirements|"
    r"qualifications)\s*:|"
    r"(?:knowledge|skills)\s+and\s+abilities\s*:"
    r")",
    re.I,
)

MULTI_ROLE_TITLE = re.compile(
    r"\b(?:positions|multiple\s+roles|"
    r"various\s+positions)\b",
    re.I,
)

DEGREE_CUE = re.compile(
    r"\b(?:BS\s+in|BA\s+in|"
    r"bachelor[\u2019\x27]?s?\s+degree)\b",
    re.I,
)


# ==================================================
# 6. NORMALIZE TEXT
# ==================================================

def normalize(text):
    """Normalize whitespace without changing terminology."""

    return " ".join(
        str(text or "").split()
    )


# ==================================================
# 7. REMOVE FOOTER
# ==================================================

def remove_footer(text):
    """
    Remove boilerplate when safe.

    Preserve responsibilities that appear
    after benefits or company information.
    """

    text = normalize(text)

    match = FOOTER_RE.search(text)

    if not match:
        return text

    before = text[:match.start()].strip()
    remainder = text[match.end():]

    later_duties = LATER_DUTIES.search(remainder)

    if later_duties:

        duties = remainder[
            later_duties.start():
        ].strip()

        if (
            ROLE_HEADING.search(before)
            or DUTIES_LEAD.search(before)
        ):
            return "\n".join(
                part
                for part in (before, duties)
                if part
            )

        return duties

    label = match.group(0).casefold()

    promotional = bool(
        re.match(
            r"(?:company\s+benefits|"
            r"employee\s+benefits|benefits|"
            r"perks|about\s+us|"
            r"about\s+the\s+company|"
            r"company\s+overview)",
            label,
        )
    )

    # Preserve uncertain descriptions
    # rather than discarding potential duties.
    if promotional and not (
        ROLE_HEADING.search(before)
        or DUTIES_LEAD.search(before)
    ):
        return text

    return before or text


# ==================================================
# 8. SELECT RELEVANT DESCRIPTION
# ==================================================

def select_description(description, title=""):

    raw = normalize(description)

    if not raw:
        return "", "empty"

    cleaned = remove_footer(raw)

    if MULTI_ROLE_TITLE.search(title):
        return (
            cleaned,
            "multiple roles - review",
        )

    headings = (
        list(ROLE_HEADING.finditer(cleaned))
        + list(DUTIES_LEAD.finditer(cleaned))
        + list(re.finditer(r"\bour\s+(?:[\w/&-]+\s+){1,12}will\s*:", cleaned, re.I))
    )

    if headings:

        first = min(
            headings,
            key=lambda match: match.start()
        )

        return (
            cleaned[first.start():].strip(),
            "role section",
        )

    return (
        cleaned,
        "unsectioned - review",
    )


# ==================================================
# 9. BUILD ESCO MATCHER
# ==================================================

def build_matcher():

    if not ESCO_FILE.is_file():

        raise FileNotFoundError(
            f"ESCO file missing: {ESCO_FILE}"
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

        preferred = normalize(
            row.preferredLabel
        )

        if not preferred or not row.conceptUri:
            continue

        labels = [
            preferred,
            *row.altLabels.splitlines(),
        ]

        for raw_label in labels:

            label = normalize(
                raw_label
            ).casefold()

            is_alternative = (
                label != preferred.casefold()
            )

            if (
                len(label) < 3
                or label in BLOCKED_LABELS
            ):
                continue

            if (
                is_alternative
                and label in BLOCKED_ALTERNATIVES
            ):
                continue

            if (
                is_alternative
                and " " not in label
                and label not in ALLOWED_ACRONYM_ALIASES
            ):
                continue

            label_map[label][
                row.conceptUri
            ] = preferred

    matcher = KeywordProcessor(
        case_sensitive=False
    )

    for label, candidates in label_map.items():

        if len(candidates) != 1:
            continue

        uri, skill_name = next(
            iter(candidates.items())
        )

        matcher.add_keyword(
            label,
            (uri, skill_name, label),
        )

    root_rows = skills[
        skills["preferredLabel"].str.casefold()
        == "root cause analysis"
    ]

    root_uri = (
        root_rows.iloc[0].conceptUri
        if len(root_rows) == 1
        else None
    )

    print(
        f"ESCO patterns: {len(matcher):,}"
    )

    return matcher, root_uri


# ==================================================
# 10. DEGREE CONTEXT
# ==================================================

def degree_mention_only(text, label):

    mentions = list(
        re.finditer(
            r"\b" + re.escape(label) + r"\b",
            text,
            re.I,
        )
    )

    if not mentions:
        return False

    return all(
        bool(
            DEGREE_CUE.search(
                re.split(
                    r"[.;!?\n]",
                    text[
                        max(0, hit.start() - 160):
                        hit.start()
                    ],
                )[-1]
            )
        )
        for hit in mentions
    )


# ==================================================
# 11. MATCH SKILLS
# ==================================================

def match_skills(
    text,
    title,
    matcher,
    root_uri,
):

    found = {}

    context = f"{title}\n{text}"

    for uri, name, label in matcher.extract_keywords(text):

        # Filter figurative expression:
        # "the fine art of customer service"
        if label == "fine art":

            matches = list(
                re.finditer(
                    r"\bfine art\b",
                    text,
                    re.I,
                )
            )

            if matches and all(
                re.match(
                    r"\s+of\b",
                    text[match.end():],
                    re.I,
                )
                for match in matches
            ):
                continue

        # Avoid degree majors treated as skills
        # in selected contexts.
        if label in {
            "journalism",
            "economics",
        }:

            if degree_mention_only(text, label):
                continue

        rule = CONTEXT_RULES.get(label)

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

    if root_uri and ROOT_CAUSE_RE.search(text):

        found[root_uri] = {
            "skill": "root cause analysis",
            "matched_label": "root cause (context rule)",
            "esco_uri": root_uri,
        }

    return list(found.values())


# ==================================================
# 12. PREDICT SKILLS
# ==================================================

def predict(row, matcher, root_uri):

    title = normalize(
        row.get("Title", "")
    )

    requirements = remove_footer(
        row.get("Requirements", "")
    )

    # Remove promotional statistics text from requirements.
    promo = re.search(
        r"if you meet the requirements of this position.{0,180}following statistics",
        requirements,
        re.IGNORECASE | re.DOTALL,
    )

    if promo:
        requirements = requirements[:promo.start()].strip()

    if GENERIC_REQUIREMENTS.search(requirements):
        requirements = ""

    description, desc_type = select_description(
        row.get("Description", ""),
        title,
    )

    # Important fix:
    # Do not ignore the description just
    # because requirements have a matched skill.

    if description and desc_type == "role section":

        source = (
            "requirements + role duties"
            if requirements
            else "role duties"
        )

    elif (
        description
        and desc_type == "multiple roles - review"
    ):

        source = "multiple roles - review"

    elif description:

        source = (
            "requirements + unsectioned description - review"
            if requirements
            else "unsectioned description - review"
        )

    else:

        source = (
            "requirements"
            if requirements
            else "empty"
        )

    text = "\n".join(
        part
        for part in [
            requirements,
            description,
        ]
        if part
    )

    if desc_type == "unsectioned - review" and match_skills(requirements, title, matcher, root_uri):
        text = requirements
        source = "requirements (unsectioned description skipped - review)"

    extracted = match_skills(
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
            skill["skill"]
            for skill in extracted
        ),

        "ESCO_URIs": "; ".join(
            skill["esco_uri"]
            for skill in extracted
        ),

        "MatchedLabels": "; ".join(
            skill["matched_label"]
            for skill in extracted
        ),

        "ReviewFlag": (
            "CHECK_UNSECTIONED_TEXT"
            if "review" in source
            else ""
        ),

        "TextForReview": text,
    }


# ==================================================
# 13. COMPARE WITH V6
# ==================================================

def compare_to_v6(rows):

    if not V6_PREDICTIONS.exists():

        print(
            "V6 predictions not found. "
            "Comparison skipped."
        )
        return

    old = pd.read_csv(
        V6_PREDICTIONS,
        dtype=str,
        keep_default_na=False,
    )

    old_by_key = {
        (row.JobID, row.WindowID): row
        for row in old.itertuples(index=False)
    }

    changes = []

    for new in rows:

        key = (
            new["JobID"],
            new["WindowID"],
        )

        before = old_by_key.get(key)

        if before is None:
            continue

        old_skills = {
            skill.strip()
            for skill in before.ExtractedSkills.split(";")
            if skill.strip()
        }

        new_skills = {
            skill.strip()
            for skill in new["ExtractedSkills"].split(";")
            if skill.strip()
        }

        if old_skills == new_skills:
            continue

        changes.append({
            "JobID": new["JobID"],
            "WindowID": new["WindowID"],
            "Title": new["Title"],

            "Removed": "; ".join(
                sorted(old_skills - new_skills)
            ),

            "Added": "; ".join(
                sorted(new_skills - old_skills)
            ),

            "ReviewFlag": new["ReviewFlag"],
        })

    columns = [
        "JobID",
        "WindowID",
        "Title",
        "Removed",
        "Added",
        "ReviewFlag",
    ]

    COMPARE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        changes,
        columns=columns,
    ).to_csv(
        COMPARE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Changed versus V6: "
        f"{len(changes)} / {len(rows)}"
    )

    print(
        f"Comparison: {COMPARE_OUTPUT}"
    )


# ==================================================
# 14. RUN SAMPLE
# ==================================================

def run_sample(args, matcher, root_uri):

    data = pd.read_csv(
        args.sample_path,
        dtype=str,
        keep_default_na=False,
        usecols=INPUT_COLUMNS,
    )

    rows = [
        predict(row, matcher, root_uri)
        for row in data.to_dict("records")
    ]

    target = args.output or SAMPLE_OUTPUT

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    ).to_csv(
        target,
        index=False,
        encoding="utf-8-sig",
    )

    with_skills = sum(
        bool(row["ExtractedSkills"])
        for row in rows
    )

    flagged = sum(
        bool(row["ReviewFlag"])
        for row in rows
    )

    print(
        f"Tested: {len(rows)} jobs"
    )

    print(
        f"Jobs with extracted skills: {with_skills}"
    )

    print(
        f"Flagged for review: {flagged}"
    )

    print(
        f"Saved: {target}"
    )

    if (
        args.sample_path.resolve()
        == DEFAULT_SAMPLE.resolve()
    ):
        compare_to_v6(rows)


# ==================================================
# 15. FULL EXTRACTION
# ==================================================

def run_full(args, matcher, root_uri):

    if not JOBS_FILE.exists():
        raise FileNotFoundError(
            JOBS_FILE
        )

    allowed_windows = (
        {
            window.strip()
            for window in args.windows.split(",")
            if window.strip()
        }
        if args.windows
        else None
    )

    target = args.output or FULL_OUTPUT

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        col
        for col in OUTPUT_COLUMNS
        if col != "TextForReview"
    ]

    processed = 0

    with target.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
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

            if allowed_windows is not None:

                chunk = chunk[
                    chunk["WindowID"].isin(
                        allowed_windows
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

            if processed and processed % 50000 == 0:

                print(
                    f"Processed {processed:,}",
                    flush=True,
                )

    print(
        f"Processed {processed:,} jobs. "
        f"Saved: {target}"
    )


# ==================================================
# 16. SELF-TEST
# ==================================================

def self_test():
    """Fast tests without loading ESCO taxonomy."""

    text = remove_footer(
        "Benefits include insurance. "
        "Responsibilities: PLC programming"
    )

    assert "PLC programming" in text

    text = remove_footer(
        "Responsibilities: audit reports. "
        "Benefits include insurance. "
        "Duties: use Excel"
    )

    assert "audit reports" in text
    assert "use Excel" in text

    text = remove_footer(
        "Company sells food containers. "
        "Benefits include insurance. "
        "This position is responsible for "
        "PLC programming and blueprints"
    )

    assert "PLC programming" in text
    assert "food containers" not in text

    class FakeMatcher:

        def extract_keywords(self, text):

            skills = {
                "accounting": "accounting",
                "auditing": "auditing",
                "excel": "excel",
            }

            return [
                (
                    "uri:" + skill,
                    skill,
                    skill,
                )
                for skill in skills
                if re.search(
                    r"\b" + re.escape(skill) + r"\b",
                    text,
                    re.I,
                )
            ]

    row = {
        "JobID": "1",
        "WindowID": "1",
        "Title": "Auditor",
        "Requirements": "Accounting",
        "Description": (
            "Responsibilities: Auditing, Excel and financial reporting"
        ),
    }

    result = predict(
        row,
        FakeMatcher(),
        None,
    )

    found = result["ExtractedSkills"]

    assert "accounting" in found
    assert "auditing" in found
    assert "excel" in found

    print(
        "Self-test passed: footer handling "
        "and requirements + description extraction"
    )


# ==================================================
# 17. MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="TalentMatch ESCO Skills Extraction"
    )

    parser.add_argument(
        "--full",
        action="store_true",
    )

    parser.add_argument(
        "--windows",
        help="Comma-separated WindowIDs (use 1 for our project)",
    )

    parser.add_argument(
        "--all-windows",
        action="store_true",
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

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.windows and args.all_windows:
        parser.error(
            "Use either --windows or --all-windows"
        )

    if args.full and not (
        args.windows or args.all_windows
    ):
        parser.error(
            "Full extraction requires --windows"
        )

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
