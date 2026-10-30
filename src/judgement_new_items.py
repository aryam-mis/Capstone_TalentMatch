# src/judgement_new_items.py
# Blind grading sheet for the re-ranker recommendations that have not been judged yet, plus a separate key
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--recs", default="results/judgement/reranker_top10.csv")
ap.add_argument("--comparison", required=True)
ap.add_argument("--jobs", default="data/processed/jobs_clean.tsv")
ap.add_argument("--out", default="results/judgement/reranker_new_items.csv")
ap.add_argument("--key", default="results/judgement/reranker_new_items_key.csv")
a = ap.parse_args()

recs = pd.read_csv(a.recs)
new = recs[~recs["already_graded"]].copy()
cv = pd.read_csv(a.comparison).drop_duplicates("cv_id")[["cv_id", "cv_profile"]]

cols = list(pd.read_csv(a.jobs, sep="\t", nrows=0).columns)
desc = next((c for c in cols if c.lower().startswith("descr")), None)
req = next((c for c in cols if c.lower().startswith("requir")), None)
if desc is None or req is None:
    raise SystemExit(f"description or requirements column not found in {a.jobs}: {cols}")
print("job text columns:", desc, req)

keys = new[["JobID", "WindowID"]].drop_duplicates()
found = []
for chunk in pd.read_csv(a.jobs, sep="\t", usecols=["JobID", "WindowID", desc, req], chunksize=200000):
    hit = chunk.merge(keys, on=["JobID", "WindowID"])
    if len(hit):
        found.append(hit)
texts = pd.concat(found).drop_duplicates(["JobID", "WindowID"])

sheet = new.merge(texts, on=["JobID", "WindowID"], how="left").merge(cv, on="cv_id", how="left")
sheet = sheet.drop(columns=["item_id"]).sample(frac=1, random_state=42).reset_index(drop=True)
sheet.insert(0, "item_id", [f"RR-{i:03d}" for i in range(1, len(sheet) + 1)])

key = sheet[["item_id", "cv_id", "JobID", "WindowID", "rank", "score"]]
key.to_csv(a.key, index=False, encoding="utf-8-sig")

sheet = sheet.rename(columns={"Title": "job_title", "City": "job_city", "State": "job_state",
                              desc: "job_description", req: "job_requirements"})
for c in ["reviewer1_score", "reviewer1_notes", "reviewer2_score", "reviewer2_notes"]:
    sheet[c] = ""
blind = ["item_id", "cv_id", "cv_profile", "job_title", "job_description", "job_requirements",
         "job_city", "job_state", "reviewer1_score", "reviewer1_notes", "reviewer2_score", "reviewer2_notes"]
sheet[blind].to_csv(a.out, index=False, encoding="utf-8-sig")
print("saved", a.out, len(sheet), "items | without job text:", int(sheet["job_description"].isna().sum()))
print("key saved", a.key)
