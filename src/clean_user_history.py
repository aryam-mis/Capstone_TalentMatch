from pathlib import Path
import pandas as pd
import html
import re
import csv

PROJECT = Path(__file__).resolve().parent.parent
INPUT = PROJECT / "data/user_history.tsv"
OUTPUT = PROJECT / "data/processed/user_history_clean.tsv"

def clean_title(value):
    if pd.isna(value):
        return value

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

total_rows = 0
missing_before = 0
missing_after = 0

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

for i, chunk in enumerate(
    pd.read_csv(
        INPUT,
        sep="\t",
        dtype=str,
        chunksize=10000,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE
    )
):
    missing_before += chunk["JobTitle"].eq("").sum()

    chunk["JobTitle"] = chunk["JobTitle"].apply(clean_title)

    missing_after += chunk["JobTitle"].eq("").sum()

    chunk.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
        mode="w" if i == 0 else "a",
        header=i == 0
    )

    total_rows += len(chunk)

    if total_rows % 100000 == 0:
        print(f"Processed {total_rows:,} rows", flush=True)

print("\nCleaning completed!")
print("Total rows:", total_rows)
print("Missing titles before:", missing_before)
print("Missing titles after:", missing_after)
print("Saved to:", OUTPUT)