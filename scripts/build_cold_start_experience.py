import re
import shutil
from pathlib import Path

import pandas as pd

SOURCE = Path("data/processed/jobs_clean.tsv")
OUTPUT = Path("artifacts/cold_start_experience.parquet")

PATTERNS = [
    # 3-5 years, 5 to 8+ years, 5 or more years
    re.compile(
        r"\b(\d{1,2})\s*(?:-|–|to|or\s+more|\+)\s*"
        r"(?:\d{1,2}\+?\s*)?(?:years?|yrs?)\b",
        re.I,
    ),
    # 5 years experience, 2 years UI design experience
    re.compile(
        r"\b(\d{1,2})\s*(?:years?|yrs?)\b"
        r"(?=[^\n.!?]{0,80}\b(?:experience|exp|working|background)\b)",
        re.I,
    ),
    # Minimum of 5 years, at least 3 years
    re.compile(
        r"\b(?:minimum(?:\s+of)?|at\s+least)\s+"
        r"(\d{1,2})\s*(?:years?|yrs?)\b",
        re.I,
    ),
]


def extract_years(text):
    if pd.isna(text):
        return None

    text = str(text)
    values = []

    for pattern in PATTERNS:
        values.extend(int(x) for x in pattern.findall(text))

    return min(values) if values else None


def process_chunk(chunk):
    text = (
        chunk["Requirements"].fillna("").astype(str)
        + " "
        + chunk["Description"].fillna("").astype(str)
    )

    return pd.DataFrame({
        "JobID": chunk["JobID"],
        "WindowID": chunk["WindowID"],
        "required_years": text.map(extract_years),
        "lead_qa": chunk["Description"].fillna("").str.contains(
            r"^\s*Lead\s+QA\s+Engineer", case=False, regex=True
        ),
    })


if __name__ == "__main__":
    tests = {
        "3-5 Years": 3,
        "Minimum 5 to 8+ years of SAP experience": 5,
        "5 or more years of experience": 5,
        "At least 2-4 years UI Design experience": 2,
        "Minimum of 5 years UX experience": 5,
        "Four year college degree": None,
    }

    for text, expected in tests.items():
        actual = extract_years(text)
        assert actual == expected, (text, actual, expected)

    print("Regex tests passed.")

    chunks = pd.read_csv(
        SOURCE,
        sep="\t",
        usecols=["JobID", "WindowID", "Requirements", "Description"],
        chunksize=50000,
    )

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(process_chunk(chunk))
        print(f"Processed chunk {i}", flush=True)

    result = pd.concat(parts, ignore_index=True)

    assert not result.duplicated(["JobID", "WindowID"]).any()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        shutil.copy2(OUTPUT, OUTPUT.with_suffix(".parquet.bak"))

    result.to_parquet(OUTPUT, index=False)

    print("Total jobs:", len(result))
    print("Experience detected:", result["required_years"].notna().sum())
    print("Saved:", OUTPUT)
