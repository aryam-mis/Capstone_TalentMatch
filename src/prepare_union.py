import pandas as pd

P = "data/processed/"
OPEN_FROM = pd.Timestamp("2012-05-30")   # earliest period start (re-ranker training period)
OPEN_TO = pd.Timestamp("2012-06-27")     # end of test period (exclusive)

cols = ["JobID", "WindowID", "State", "Title", "Requirements", "StartDate", "EndDate"]
parts, offset = [], 0
for ch in pd.read_csv(P + "jobs_clean.tsv", sep="\t", usecols=cols,
                      chunksize=100_000, low_memory=False, quoting=3):
    ch = ch.copy()
    ch["row_id"] = range(offset, offset + len(ch))
    offset += len(ch)
    ch["StartDate"] = pd.to_datetime(ch["StartDate"], errors="coerce")
    ch["EndDate"] = pd.to_datetime(ch["EndDate"], errors="coerce")
    keep = ((ch["EndDate"].isna() | (ch["EndDate"] >= OPEN_FROM)) &
            (ch["StartDate"].isna() | (ch["StartDate"] < OPEN_TO)))
    ch = ch[keep]

    t = ch["Title"].fillna("").astype(str).str.strip()
    r = ch["Requirements"].fillna("").astype(str).str.strip()
    ch["txt_title_req"] = (t + ". " + r.str.slice(0, 600)).str.lower()   # same recipe as E2
    ch["State"] = ch["State"].fillna("JOB_NA").astype(str).str.strip().str.upper()
    parts.append(ch[["row_id", "JobID", "WindowID", "State",
                     "StartDate", "EndDate", "txt_title_req"]])

jobs = pd.concat(parts, ignore_index=True)
assert offset == 1_092_096, offset
print("jobs open in the union period:", len(jobs))
jobs.to_parquet("data/colab/jobs_open_union.parquet", index=False)