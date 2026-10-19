from pathlib import Path
import pandas as pd

# Project paths
PROJECT = Path(__file__).resolve().parent.parent
INPUT = PROJECT / "data/users.tsv"
OUTPUT = PROJECT / "data/processed/users_clean.tsv"

# Load all users
users = pd.read_csv(
    INPUT,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# Clean text columns
text_columns = [
    "City",
    "State",
    "Country",
    "DegreeType",
    "Major",
    "CurrentlyEmployed",
    "ManagedOthers"
]

for col in text_columns:
    users[col] = users[col].str.strip()

# Save cleaned users
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

users.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

# Validate output
cleaned = pd.read_csv(
    OUTPUT,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Original Rows:", len(users))
print("Cleaned Rows:", len(cleaned))

print(
    "IDs unchanged:",
    users[["UserID", "WindowID", "Split"]].equals(
        cleaned[["UserID", "WindowID", "Split"]]
    )
)

print("Cleaning completed!")