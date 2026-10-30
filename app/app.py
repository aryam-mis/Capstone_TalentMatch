# app/app.py  (Job seeker: free-text profile -> re-ranker v2 | Recruiter: RecruiterIndex, content-only)
import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from profile_recommender import ProfileRecommender  # noqa: E402
from recruiter_candidates import RecruiterIndex  # noqa: E402

st.set_page_config(page_title="TalentMatch", layout="wide")

DEGREES = ["(not specified)", "High School", "Vocational", "Associate's",
           "Bachelor's", "Master's", "PhD"]
K = 10


@st.cache_resource
def get_rec():
    return ProfileRecommender()


@st.cache_resource
def get_index():
    return RecruiterIndex()


@st.cache_resource
def get_users():
    return pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                       usecols=["UserID", "WindowID", "DegreeType", "Major",
                                "TotalYearsExperience"])


rec = get_rec()
index = get_index()
users_all = get_users()
states = sorted(rec.jobs.State.dropna().unique().tolist())


def job_line(jid, wid):
    j = rec.jobs[(rec.jobs.JobID == jid) & (rec.jobs.WindowID == wid)]
    if j.empty:
        return "Job not found among the open jobs."
    j = j.iloc[0]
    return f"{j.Title} | {j.City.title()}, {j.State}"


def candidates_for_job(jid, wid, k=K):
    top = index.top_users(jid, wid, k=k)
    top = top.merge(users_all, on=["UserID", "WindowID"], how="left")
    top["score"] = top["score"].round(3)
    top["Why"] = "-"
    return top[["rank", "UserID", "DegreeType", "Major", "TotalYearsExperience", "score", "Why"]]


st.title("TalentMatch")
st.caption("Open jobs of the test period (2012-06-18 onward). A typed profile is a new user with no "
           "application history, so the collaborative signal is empty; ranking uses the LightGBM re-ranker.")

tab1, tab2 = st.tabs(["Job seeker", "Recruiter"])

with tab1:
    c1, c2, c3 = st.columns(3)
    degree = c1.selectbox("Degree", DEGREES, index=4)
    major = c2.text_input("Major", "Business Administration")
    years = c3.number_input("Years of experience", min_value=0, max_value=40, value=5)
    roles_text = st.text_area("Past roles (one per line)",
                              "Customer Service Representative\nSales Associate")
    c4, c5 = st.columns(2)
    state = c4.selectbox("State", states, index=states.index("CA") if "CA" in states else 0)
    city = c5.text_input("City (optional)", "Los Angeles")

    if st.button("Recommend jobs"):
        roles = [r for r in roles_text.splitlines() if r.strip()]
        deg = "" if degree == DEGREES[0] else degree
        t = time.time()
        res = rec.recommend(deg, major, years, roles, state, city, k=K)
        ms = (time.time() - t) * 1000
        out = res.copy()
        out["score"] = out["score"].round(3)
        out["City"] = out["City"].str.title()
        out["Why"] = "-"   # to be filled by the matched-skills explanation
        st.dataframe(out[["rank", "Title", "City", "State", "score", "Why", "JobID", "WindowID"]],
                     width="stretch", hide_index=True)
        st.caption(f"response time: {ms:.0f} ms | window {res.attrs['window']} | "
                   f"query text: {res.attrs['text']}")

with tab2:
    top_job = rec.jobs.sort_values("job_pop").iloc[-1]
    c1, c2 = st.columns(2)
    jid = c1.number_input("JobID", min_value=0, value=int(top_job.JobID), step=1)
    wid = c2.number_input("WindowID", min_value=0, value=int(top_job.WindowID), step=1)
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