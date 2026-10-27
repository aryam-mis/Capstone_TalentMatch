import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load, rank  # noqa: E402
from recruiter_candidates import RecruiterIndex  # noqa: E402

ARTIFACT = "artifacts/reranker_v0"
FEATS = "data/processed/features_val_sample.parquet"
K = 10

st.set_page_config(page_title="TalentMatch", layout="wide")


@st.cache_resource
def get_model():
    return load(ARTIFACT)


@st.cache_resource
def get_index():
    return RecruiterIndex()


@st.cache_resource
def get_tables():
    jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                       usecols=["JobID", "WindowID", "Title", "City", "State"])
    users = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                        usecols=["UserID", "WindowID", "DegreeType", "Major",
                                 "TotalYearsExperience"])
    return jobs, users


@st.cache_resource
def get_sample():
    f = pd.read_parquet(FEATS)
    keys = f[["UserID", "WindowID"]].drop_duplicates()
    hist = pd.read_csv("data/processed/user_history_clean.tsv", sep="\t", dtype=str,
                       on_bad_lines="skip",
                       usecols=["UserID", "WindowID", "Sequence", "JobTitle"])
    for c in ["UserID", "WindowID", "Sequence"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist = hist.dropna(subset=["UserID", "WindowID", "Sequence"])
    hist = hist.astype({"UserID": "int64", "WindowID": "int64"})
    hist = hist.merge(keys, on=["UserID", "WindowID"], how="inner")
    hist = hist.sort_values(["UserID", "Sequence"])
    return f, hist


booster, meta = get_model()
jobs_all, users_all = get_tables()
feats, hist = get_sample()


def profile_text(uid):
    u = users_all[users_all.UserID == uid]
    if u.empty:
        return "No profile found."
    u = u.iloc[0]
    major = u.Major if pd.notna(u.Major) else "-"
    titles = hist[hist.UserID == uid].JobTitle.head(5).tolist()
    return (f"{u.DegreeType} | major: {major} | {u.TotalYearsExperience:g} years of experience | "
            f"previous titles (file order): {', '.join(titles) if titles else '-'}")


def jobs_for_user(uid, k=K):
    cand = feats[feats.UserID == uid]
    top = rank(booster, meta, cand, k).reset_index(drop=True)
    top.insert(0, "rank", range(1, len(top) + 1))
    top = top.merge(jobs_all, on=["JobID", "WindowID"], how="left")
    top["score"] = top["score"].round(3)
    top["Why"] = "-"   # to be filled by the skills-matching explanation
    return top[["rank", "Title", "City", "State", "score", "Why", "JobID", "WindowID"]]


def job_line(jid, wid):
    j = jobs_all[(jobs_all.JobID == jid) & (jobs_all.WindowID == wid)]
    if j.empty:
        return "Job not found in the jobs table."
    j = j.iloc[0]
    return f"{j.Title} | {j.City}, {j.State}"


def candidates_for_job(jid, wid, k=K):
    top = get_index().top_users(jid, wid, k=k)
    top = top.merge(users_all, on=["UserID", "WindowID"], how="left")
    top["score"] = top["score"].round(3)
    top["Why"] = "-"
    return top[["rank", "UserID", "DegreeType", "Major", "TotalYearsExperience", "score", "Why"]]


st.title("TalentMatch")
st.caption("Demo build: the job seeker view uses a smoke-test LightGBM model on a val sample "
           "(not final results). Scores there are comparable only within one list.")

tab1, tab2 = st.tabs(["Job seeker", "Recruiter"])

with tab1:
    uids = sorted(feats.UserID.unique().tolist())
    uid = st.selectbox("UserID (demo sample)", uids)
    st.caption(profile_text(int(uid)))
    if st.button("Recommend jobs"):
        t = time.time()
        out = jobs_for_user(int(uid))
        ms = (time.time() - t) * 1000
        st.dataframe(out, width="stretch", hide_index=True)
        st.caption(f"response time: {ms:.0f} ms")

with tab2:
    top_job = feats.groupby(["JobID", "WindowID"]).size().idxmax()
    c1, c2 = st.columns(2)
    jid = c1.number_input("JobID", min_value=0, value=int(top_job[0]), step=1)
    wid = c2.number_input("WindowID", min_value=0, value=int(top_job[1]), step=1)
    st.caption("Candidates are ranked by content similarity (bge-small cosine) between the job "
               "and each user profile, within the job's window and state. Content only: no "
               "collaborative signal, and the LightGBM re-ranker is not applied in this direction.")
    if st.button("Find candidates"):
        t = time.time()
        try:
            out = candidates_for_job(int(jid), int(wid))
            ms = (time.time() - t) * 1000
            st.write(job_line(int(jid), int(wid)))
            st.dataframe(out, width="stretch", hide_index=True)
            st.caption(f"response time: {ms:.0f} ms")
        except Exception as e:
            st.warning(f"No candidates for this job (is it in the open-jobs index?). {e}")