import pandas as pd

# validation period bounds (other periods pass their own dates)
VAL_START, VAL_END = "2012-06-06", "2012-06-14"


def open_jobs(jobs, start, end):
    """Return the jobs open during [start, end).
    Open = (EndDate missing or >= start) and (StartDate missing or < end).
    """
    j = jobs.copy()
    j["StartDate"] = pd.to_datetime(j["StartDate"], errors="coerce")
    j["EndDate"] = pd.to_datetime(j["EndDate"], errors="coerce")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = ((j["EndDate"].isna() | (j["EndDate"] >= start)) &
            (j["StartDate"].isna() | (j["StartDate"] < end)))
    return j[mask]