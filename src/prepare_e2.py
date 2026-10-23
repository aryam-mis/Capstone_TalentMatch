import pandas as pd

P = "data/processed/"
VAL_START = pd.Timestamp("2012-06-06")
VAL_END = pd.Timestamp("2012-06-14")
cols = ["JobID", "WindowID", "State", "Title", "Description", "Requirements",
        "StartDate", "EndDate"]

parts, offset = [], 0
for ch in pd.read_csv(P + "jobs_clean.tsv", sep="\t", usecols=cols,
                      chunksize=100_000, low_memory=False, quoting=3):
    ch = ch.copy()
    ch["row_id"] = range(offset, offset + len(ch))   # position in jobs_keys.parquet
    offset += len(ch)

    end = pd.to_datetime(ch["EndDate"], errors="coerce")
    start = pd.to_datetime(ch["StartDate"], errors="coerce")
    keep = (end.isna() | (end >= VAL_START)) & (start.isna() | (start < VAL_END))
    ch = ch[keep]

    t = ch["Title"].fillna("").astype(str).str.strip()
    d = ch["Description"].fillna("").astype(str).str.strip()
    r = ch["Requirements"].fillna("").astype(str).str.strip()
    ch["txt_title"] = t.str.lower()
    ch["txt_title_req"] = (t + ". " + r.str.slice(0, 600)).str.lower()
    ch["txt_full"] = (t + ". " + d.str.slice(0, 600) + ". " + r.str.slice(0, 600)).str.lower()
    ch["State"] = ch["State"].fillna("JOB_NA").astype(str).str.strip().str.upper()
    parts.append(ch[["row_id", "JobID", "WindowID", "State",
                     "txt_title", "txt_title_req", "txt_full"]])

jobs_open = pd.concat(parts, ignore_index=True)
print("open jobs:", len(jobs_open), "(expected 332650)")
assert offset == 1_092_096
jobs_open.to_parquet("data/colab/jobs_open_variants.parquet", index=False)