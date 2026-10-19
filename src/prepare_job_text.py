from pathlib import Path
import pandas as pd
import csv

PROJECT = Path(__file__).resolve().parent.parent

INPUT = PROJECT / "data/processed/jobs_text_cleaned.tsv"
OUTPUT = PROJECT / "data/processed/jobs_clean.tsv"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

total_rows = 0

for i, chunk in enumerate(
    pd.read_csv(
        INPUT,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=5000
    )
):
    # Combine job information
    chunk["text_clean"] = (
        chunk["Title"] + " " +
        chunk["Description"] + " " +
        chunk["Requirements"]
    ).str.replace(r"[\t\r\n]+", " ", regex=True)

    chunk["text_clean"] = (
        chunk["text_clean"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Save all original columns + text_clean
    chunk.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
        mode="w" if i == 0 else "a",
        header=i == 0,
        quoting=csv.QUOTE_MINIMAL
    )

    total_rows += len(chunk)

    if total_rows % 100000 == 0:
        print(f"Processed {total_rows:,} jobs", flush=True)

print("\nPreparation completed!")
print("Total jobs:", total_rows)
print("Saved to:", OUTPUT)