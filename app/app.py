# app/app.py  (skeleton: placeholder backend = popular_state_open; the re-ranker replaces it later)
import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TalentMatch", layout="wide")

RECS = "results/recs_popular_state_open.csv"   # placeholder backend
K = 10


@st.cache_data
def load():
    recs = pd.read_csv(RECS)
    jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                       usecols=["JobID", "WindowID", "Title", "City", "State"])
    keys = recs[["JobID", "WindowID"]].drop_duplicates()
    jobs = jobs.merge(keys, on=["JobID", "WindowID"], how="inner")
    return recs, jobs


recs, jobs = load()


def jobs_for_user(uid, k=K):
    r = recs[recs.UserID == uid].sort_values("rank").head(k)
    r = r.merge(jobs, on=["JobID", "WindowID"], how="left")
    r["Why"] = "-"   # to be filled by the skills-matching explanation
    return r[["rank", "Title", "City", "State", "Why", "JobID", "WindowID"]]


def users_for_job(jid, wid, k=K):
    r = recs[(recs.JobID == jid) & (recs.WindowID == wid)].sort_values("rank").head(k)
    r = r.assign(Why="-")
    return r[["UserID", "rank", "Why"]]


st.title("TalentMatch")
st.caption("Skeleton: the placeholder recommender is 'popular open jobs in the user's state'. "
           "The LightGBM re-ranker replaces it.")

tab1, tab2 = st.tabs(["Job seeker", "Recruiter"])

with tab1:
    uid = st.number_input("UserID", min_value=0, value=int(recs.UserID.iloc[0]), step=1)
    if st.button("Recommend jobs"):
        t = time.time()
        out = jobs_for_user(int(uid))
        ms = (time.time() - t) * 1000
        if out.empty:
            st.warning("No recommendations for this user.")
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
        st.caption(f"response time: {ms:.0f} ms")

with tab2:
    first = recs.iloc[0]
    c1, c2 = st.columns(2)
    jid = c1.number_input("JobID", min_value=0, value=int(first.JobID), step=1)
    wid = c2.number_input("WindowID", min_value=0, value=int(first.WindowID), step=1)
    if st.button("Find candidates"):
        t = time.time()
        out = users_for_job(int(jid), int(wid))
        ms = (time.time() - t) * 1000
        if out.empty:
            st.warning("No candidates for this job.")
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
        st.caption(f"response time: {ms:.0f} ms")