import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("src").resolve()))

from profile_recommender import ProfileRecommender as Original
from profile_recommender_experiment import ProfileRecommender as Experimental

profiles = pd.read_csv("eval/cold_start/test_profiles.csv").fillna("")
results = []

for version, Recommender in [
    ("original", Original),
    ("experimental", Experimental),
]:
    print(f"\nLoading {version} recommender...", flush=True)
    rec = Recommender()

    for _, p in profiles.iterrows():
        profile_id = str(p["profile_id"])
        roles = str(p["roles"]).split(";")

        recommendations = rec.recommend(
            degree="Bachelor's",
            major=str(p["major"]),
            years=float(p["years"]),
            roles=roles,
            state=str(p["state"]),
            city=str(p["city"]),
            k=10,
        )

        for rank, (_, job) in enumerate(recommendations.iterrows(), start=1):
            results.append({
                "profile_id": profile_id,
                "version": version,
                "major": p["major"],
                "years": p["years"],
                "requested_city": p["city"],
                "rank": rank,
                "JobID": job["JobID"],
                "WindowID": job["WindowID"],
                "Title": job["Title"],
                "City": job["City"],
                "State": job["State"],
                "content_cos": job["content_cos"],
            })

        print(f"{version} | {profile_id}: {len(recommendations)} jobs", flush=True)

output = Path("eval/cold_start/comparison_results.csv")
pd.DataFrame(results).to_csv(output, index=False)

print(f"\nSaved {len(results)} recommendations to {output}")
