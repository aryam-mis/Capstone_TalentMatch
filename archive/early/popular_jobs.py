"""
Baseline: predicts that a user will apply to the most popular jobs in
his/her city, then to the most popular jobs in his/her state.

Python 3 port. Run from the project root:
    .\\.venv\\Scripts\\python.exe src\\popular_jobs.py
"""

import csv
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(2**31 - 1)  # job descriptions can be very long

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "data" / "jobs" / "jobs.tsv"
APPS = ROOT / "data" / "apps" / "apps.tsv"
USERS = ROOT / "data" / "users.tsv"
OUT = ROOT / "results" / "popular_jobs.csv"

TOP_N = 150


def read_tsv(path, quoting=csv.QUOTE_NONE):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        for row in reader:
            yield row


print("Recording job locations...")
job_info = {}  # job_id -> [window, state, city, app_count]
for r in read_tsv(JOBS):
    try:
        job_info[r["JobID"]] = [int(r["WindowID"]), r["State"], r["City"], 0]
    except (ValueError, KeyError, TypeError):
        continue  # malformed row

print("Counting applications...")
for r in read_tsv(APPS, quoting=csv.QUOTE_MINIMAL):
    job = job_info.get(r["JobID"])
    if job is not None:
        job[3] += 1

print("Sorting jobs by popularity...")
top_city = defaultdict(list)   # (window, state, city) -> [(job_id, count)]
top_state = defaultdict(list)  # (window, state) -> [(job_id, count)]
for job_id, (window, state, city, count) in job_info.items():
    top_city[(window, state, city)].append((job_id, count))
    top_state[(window, state)].append((job_id, count))

for lst in list(top_city.values()) + list(top_state.values()):
    lst.sort(key=lambda x: x[1], reverse=True)

print("Making predictions...")
OUT.parent.mkdir(parents=True, exist_ok=True)
n_users = 0
with open(OUT, "w", encoding="utf-8", newline="") as out:
    out.write("UserId, JobIds\n")
    for u in read_tsv(USERS):
        if u["Split"] == "Train":
            continue
        window = int(u["WindowID"])
        city_jobs = top_city.get((window, u["State"], u["City"]), [])
        state_jobs = top_state.get((window, u["State"]), [])

        # city first, then state; skip duplicates
        seen, picks = set(), []
        for job_id, _ in city_jobs + state_jobs:
            if job_id not in seen:
                seen.add(job_id)
                picks.append(job_id)
            if len(picks) == TOP_N:
                break

        out.write(f"{u['UserID']},{' '.join(picks)}\n")
        n_users += 1

print(f"Done. {n_users} users -> {OUT}")