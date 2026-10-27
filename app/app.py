# app/app.py  (model-backed on the val sample; swap the features file for real candidates later)
import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load, rank  # noqa: E402

ARTIFACT = "artifacts/reranker_v0"
FEATS = "data/processed/features_val_sample.parquet"
K = 10

st.set_page_config(page_title="TalentMatch", layout="wide")


@st.cache_resource
def get_model():
    return load(ARTIFACT)


@st.cache_resource
def get_data():
    f = pd.read_parquet(FEATS)
    keys = f[["UserID", "WindowID"]].drop_duplicates()

    jobs = pd.read_csv("data/processed/jobs_clean.tsv", sep="\t",
                       usecols=["JobID", "WindowID", "Title", "City", "State"])
    jobs = jobs.merge(f[["JobID", "WindowID"]].drop_duplicates(),
                      on=["JobID", "WindowID"], how="inner")

    users = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                        usecols=["UserID", "WindowID", "DegreeType", "Major",
                                 "TotalYearsExperience"])
    users = users.merge(keys, on=["UserID", "WindowID"], how="inner")

    hist = pd.read_csv("data/processed/user_history_clean.tsv", sep="\t", dtype=str,
                       on_bad_lines="skip",
                       usecols=["UserID", "WindowID", "Sequence", "JobTitle"])
    for c in ["UserID", "WindowID", "Sequence"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist = hist.dropna(subset=["UserID", "WindowID", "Sequence"])
    hist = hist.astype({"UserID": "int64", "WindowID": "int64"})
    hist = hist.merge(keys, on=["UserID", "WindowID"], how="inner")
    hist = hist.sort_values(["UserID", "Sequence"])
    return f, jobs, users, hist


booster, meta = get_model()
feats, jobs, users, hist = get_data()


def profile_text(uid):
    u = users[users.UserID == uid]
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
    top = top.merge(jobs, on=["JobID", "WindowID"], how="left")
    top["score"] = top["score"].round(3)
    top["Why"] = "-"   # to be filled by the skills-matching explanation
    return top[["rank", "Title", "City", "State", "score", "Why", "JobID", "WindowID"]]


def users_for_job(jid, wid, k=K):
    cand = feats[(feats.JobID == jid) & (feats.WindowID == wid)]
    if cand.empty:
        return cand
    top = rank(booster, meta, cand, k).reset_index(drop=True)
    top.insert(0, "rank", range(1, len(top) + 1))
    top["score"] = top["score"].round(3)
    top["Why"] = "-"
    top = top.merge(users, on=["UserID", "WindowID"], how="left")
    return top[["rank", "UserID", "DegreeType", "Major", "TotalYearsExperience", "score", "Why"]]


st.title("TalentMatch")
st.caption("Demo on a val sample with a smoke-test model (not final results). "
           "Scores are ranking scores, comparable only within one list.")

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
    st.caption("Candidates here are users who have this job in their candidate list; "
               "scores come from lists built per user, so comparing across users is approximate.")
    if st.button("Find candidates"):
        t = time.time()
        out = users_for_job(int(jid), int(wid))
        ms = (time.time() - t) * 1000
        if out.empty:
            st.warning("No candidates for this job in the demo sample.")
        else:
            st.dataframe(out, width="stretch", hide_index=True)
        st.caption(f"response time: {ms:.0f} ms")