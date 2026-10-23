import pandas as pd

# حدود الفتح لفترة val (لـ test وفترة التدريب نمرر قيم ثانية)
VAL_START, VAL_END = "2012-06-06", "2012-06-14"


def open_jobs(jobs, start, end):
    """يرجّع الوظائف المفتوحة خلال [start, end).
    مفتوحة = (EndDate فاضي أو >= start) و (StartDate فاضي أو < end)
    """
    j = jobs.copy()
    j["StartDate"] = pd.to_datetime(j["StartDate"], errors="coerce")
    j["EndDate"] = pd.to_datetime(j["EndDate"], errors="coerce")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = ((j["EndDate"].isna() | (j["EndDate"] >= start)) &
            (j["StartDate"].isna() | (j["StartDate"] < end)))
    return j[mask]