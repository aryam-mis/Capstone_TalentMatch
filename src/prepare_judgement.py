"""Prepare blinded reviewer CSVs without models or external dependencies.

Example (run only after the inputs arrive):
    python3 src/prepare_judgement.py --pop1 recs_pop1.csv --pop0 recs_pop0.csv \
        --cvs data/cvs.csv --jobs jobs_for_review.tsv --out-dir results/judgement/run1

Jobs CSV/TSV must contain JobID, WindowID, Title, Description, Requirements,
and State (City is optional). It is streamed; only requested jobs are retained.
The output directory must not exist. Give each reviewer only their own CSV;
keep internal/ private. Scores remain blank. Top-10 is mandatory.
"""

import argparse
import csv
import hashlib
import html
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


KEYS = ("cv_id", "JobID", "WindowID")
REVIEW_FIELDS = [
    "item_id", "cv_id", "cv_state", "cv_profile", "job_title",
    "job_description", "job_requirements", "job_city", "job_state",
    "score", "notes", "status",
]


def records(path, required):
    """Read CSV/TSV as strings, preserving identifiers and leading zeros."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not set(required).issubset(fields):
            raise ValueError(f"{path}: duplicate headers or missing columns: {sorted(set(required) - set(fields))}")
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}: malformed row at CSV record {line}")
            yield {k: v.strip() for k, v in row.items()}


def clean_text(text):
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\\r", " ").replace("\\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cvs(path, expected):
    cvs = {}
    for row in records(path, ("cv_id", "state", "profile_text")):
        cv = row["cv_id"]
        if not cv or not row["state"] or not row["profile_text"] or cv in cvs:
            raise ValueError(f"CV file: empty required value or duplicate cv_id: {cv!r}")
        row["state"] = row["state"].upper()
        cvs[cv] = row
    if len(cvs) != expected:
        raise ValueError(f"CV file: expected {expected} CVs, found {len(cvs)}")
    return cvs


def load_recs(path, configuration, cvs):
    rows, issues = [], []
    ranks = {cv: [] for cv in cvs}
    seen = set()
    for row in records(path, (*KEYS, "rank", "state")):
        cv = row["cv_id"]
        if cv not in cvs:
            issues.append(f"unknown cv_id {cv!r}")
            continue
        if not row["JobID"] or not row["WindowID"]:
            issues.append(f"{cv}: empty job key")
        try:
            rank = int(row["rank"])
        except ValueError:
            issues.append(f"{cv}: invalid rank {row['rank']!r}")
            continue
        ranks[cv].append(rank)
        key = tuple(row[k] for k in KEYS)
        if key in seen:
            issues.append(f"duplicate CV-job key {key}")
        seen.add(key)
        if row["state"].upper() != cvs[cv]["state"]:
            issues.append(f"{cv}: recommendation state differs from CV state")
        # Do not propagate scores, truncated job_text, or prior grader columns.
        rows.append(dict(zip(KEYS, key), configuration=configuration, rank=rank))
    for cv, values in ranks.items():
        if sorted(values) != list(range(1, 11)):
            issues.append(f"{cv}: expected ranks 1..10 exactly once; found {len(values)} rows, ranks={sorted(values)}")
    if issues:
        raise ValueError(f"{configuration} validation failed:\n" + "\n".join(issues))
    return rows


def load_jobs(path, wanted):
    jobs = {}
    required = ("JobID", "WindowID", "Title", "Description", "Requirements", "State")
    # Scan through EOF to detect duplicate requested keys, retaining only wanted rows.
    for row in records(path, required):
        key = (row["JobID"], row["WindowID"])
        if key not in wanted:
            continue
        if key in jobs:
            raise ValueError(f"Jobs file: duplicate requested key {key}; resolve before grading")
        job = {col: clean_text(row.get(col, "")) for col in (*required, "City")}
        if not job["Title"] or not (job["Description"] or job["Requirements"]) or not job["State"]:
            raise ValueError(f"Jobs file: insufficient title/text/state for {key}")
        jobs[key] = job
    missing = wanted - jobs.keys()
    if missing:
        raise ValueError(f"Jobs file: missing {len(missing)} requested keys: {sorted(missing)}")
    return jobs


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(pop1, pop0, cvs_path, jobs_path, out_dir, expected_cvs=30, seed=42):
    csv.field_size_limit(100_000_000)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"Output directory already exists; refusing to overwrite: {out_dir}")
    if Path(pop1).resolve() == Path(pop0).resolve():
        raise ValueError("Provide two separate recommendation files")
    cvs = load_cvs(cvs_path, expected_cvs)
    recs = load_recs(pop1, "pop_weight=1.0", cvs) + load_recs(pop0, "pop_weight=0.0", cvs)
    pairs = sorted({tuple(r[k] for k in KEYS) for r in recs})
    jobs = load_jobs(jobs_path, {(j, w) for _, j, w in pairs})
    for cv, j, w in pairs:
        if jobs[j, w]["State"].upper() != cvs[cv]["state"]:
            raise ValueError(f"Job state differs from CV scenario state for {(cv, j, w)}")

    # Assign neutral IDs independently of model/rank and then shuffle each form.
    random.Random(seed).shuffle(pairs)
    ids = {key: f"ITEM-{n:04d}" for n, key in enumerate(pairs, 1)}
    items = []
    for cv, j, w in pairs:
        job = jobs[j, w]
        items.append(dict(zip(REVIEW_FIELDS, [
            ids[cv, j, w], cv, cvs[cv]["state"], cvs[cv]["profile_text"],
            job["Title"], job["Description"], job["Requirements"], job["City"],
            job["State"], "", "", "",
        ])))
    mapping = [{**r, "item_id": ids[tuple(r[k] for k in KEYS)]} for r in recs]
    manifest = {
        "expected_cvs": expected_cvs, "k": 10, "seed": seed,
        "recommendation_rows": dict(Counter(r["configuration"] for r in recs)),
        "unique_grading_items": len(items), "shared_items": len(recs) - len(items),
        "score_values": [0, 1, 2], "scores_left_blank": True,
        "inputs": {name: {"path": str(path), "sha256": fingerprint(path)}
                   for name, path in [("pop1", pop1), ("pop0", pop0), ("cvs", cvs_path)]},
        "jobs_source": {"path": str(jobs_path), "size_bytes": Path(jobs_path).stat().st_size},
        "selected_display_sha256": hashlib.sha256(
            json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "limitations": [
            "Configuration labels come from CLI inputs; weight/run provenance must be checked against Layan's run metadata.",
            "Full source columns are required, but semantic sufficiency of job text needs human review.",
            "Reviewer CSVs do not enforce allowed scores; validate returned scores before analysis.",
        ],
    }
    # No output is created until all input validation has succeeded.
    internal = out_dir / "internal"
    out_dir.mkdir(parents=True, exist_ok=False)
    internal.mkdir()
    for reviewer in (1, 2):
        form = items.copy()
        random.Random(seed + reviewer).shuffle(form)
        write_csv(out_dir / f"reviewer{reviewer}.csv", REVIEW_FIELDS, form)
    write_csv(internal / "recommendation_item_map.csv", ["configuration", *KEYS, "rank", "item_id"], mapping)
    with (internal / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pop1", required=True, type=Path)
    parser.add_argument("--pop0", required=True, type=Path)
    parser.add_argument("--cvs", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path, help="Full job texts, CSV or TSV; streamed")
    parser.add_argument("--out-dir", required=True, type=Path, help="New directory; existing paths are refused")
    parser.add_argument("--expected-cvs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.expected_cvs < 1:
            raise ValueError("--expected-cvs must be positive")
        result = prepare(args.pop1, args.pop0, args.cvs, args.jobs, args.out_dir, args.expected_cvs, args.seed)
    except (ValueError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Prepared {result['unique_grading_items']} unique items per reviewer; "
          f"{result['shared_items']} shared pairs graded once per reviewer. Scores are blank.")
    print("Share only reviewer1.csv / reviewer2.csv with the respective reviewer; keep internal/ private.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
