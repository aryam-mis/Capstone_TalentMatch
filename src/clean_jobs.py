from pathlib import Path
import pandas as pd
import re
import html
import csv

# =====================================
# 1. Project Paths
# =====================================

PROJECT = Path(__file__).resolve().parent.parent

INPUT = PROJECT / "data/jobs/jobs.tsv"

OUTPUT = PROJECT / "data/processed/jobs_text_cleaned.tsv"


# =====================================
# 2. Text Cleaning Function
# =====================================

def clean_text(value):
    if pd.isna(value):
        return ""

    text = html.unescape(str(value))

    # Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # Remove escaped line breaks
    text = text.replace("\\r", " ")
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================
# 3. Process Full Dataset in Chunks
# =====================================

total_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT,
        sep="\t",
        chunksize=5000,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE
    )
):

    # Clean text columns
    for column in ["Title", "Description", "Requirements"]:
        chunk[column] = chunk[column].apply(clean_text)

    # Save cleaned data
    chunk.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
        mode="w" if chunk_number == 0 else "a",
        header=chunk_number == 0,
        quoting=csv.QUOTE_MINIMAL
    )

    total_rows += len(chunk)

    print(f"Processed {total_rows:,} rows", flush=True)


# =====================================
# 4. Final Result
# =====================================

print("\nCleaning completed!")
print(f"Total rows processed: {total_rows:,}")
print(f"Cleaned file saved to: {OUTPUT}")