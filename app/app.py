# app/app.py  (Job seeker: free-text profile -> re-ranker v2 + Why | Recruiter: typed new job or existing job)
import html
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from profile_recommender import ProfileRecommender  # noqa: E402
from recruiter_candidates import RecruiterIndex  # noqa: E402
from why_skills import WhyMatcher  # noqa: E402
import extract_esco_skills_candidate as esco  # noqa: E402

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
def get_why():
    return WhyMatcher()


@st.cache_resource
def get_users():
    return pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                       usecols=["UserID", "WindowID", "DegreeType", "Major",
                                "TotalYearsExperience"])


@st.cache_resource
def get_history():
    """Past job titles per user (same source as the user embedding text), sorted for lookup."""
    h = pd.read_csv("data/processed/user_history_clean.tsv", sep="\t",
                    usecols=["UserID", "WindowID", "Sequence", "JobTitle"],
                    dtype={"JobTitle": str})
    h = h.dropna(subset=["JobTitle"])
    h = h[h["JobTitle"].str.strip() != ""]
    return h.sort_values(["UserID", "WindowID", "Sequence"]).reset_index(drop=True)


rec = get_rec()
index = get_index()
why = get_why()
users_all = get_users()
history = get_history()
states = sorted(rec.jobs.State.dropna().unique().tolist())


def past_roles(top, n=5):
    """First n past job titles of each candidate, joined with ' ; '."""
    h = history[history["UserID"].isin(top["UserID"])]
    h = h.merge(top[["UserID", "WindowID"]], on=["UserID", "WindowID"])
    roles = (h.groupby(["UserID", "WindowID"])["JobTitle"]
             .apply(lambda s: " ; ".join(s.str.strip().head(n))).rename("Past roles").reset_index())
    return top.merge(roles, on=["UserID", "WindowID"], how="left")


def typed_job_skills(title, requirements):
    """ESCO skills of a job typed in the recruiter tab: {esco_uri: info}."""
    t = esco.normalize(title)
    found = esco.match_skills(f"{title}. {requirements}", t, why.matcher, why.root)
    return {s["esco_uri"]: s for s in found}


def recruiter_why(top, job_sk):
    """ESCO skills of the job that also appear in the candidate's major or past roles."""
    out = []
    for major, roles in zip(top["Major"], top["Past roles"]):
        text = ". ".join(x for x in (major, roles) if isinstance(x, str) and x.strip())
        cand = why.cv_skills(text)
        shared = [job_sk[u]["skill"] for u in cand if u in job_sk]
        out.append(", ".join(shared) if shared else "-")
    return out


def job_line(jid, wid):
    j = rec.jobs[(rec.jobs.JobID == jid) & (rec.jobs.WindowID == wid)]
    if j.empty:
        return "Job not found among the open jobs."
    j = j.iloc[0]
    return f"{j.Title} | {j.City.title()}, {j.State}"


def show_candidates(top, ms, job_sk):
    t = time.time()
    top = top.merge(users_all, on=["UserID", "WindowID"], how="left")
    top["Major"] = top["Major"].map(lambda m: html.unescape(m) if isinstance(m, str) else m)
    top = past_roles(top)
    top["score"] = top["score"].round(3)
    top["Why"] = recruiter_why(top, job_sk)
    ms_why = (time.time() - t) * 1000
    st.dataframe(top[["rank", "UserID", "DegreeType", "Major", "TotalYearsExperience",
                      "Past roles", "score", "Why"]],
                 width="stretch", hide_index=True)
    st.caption(f"response time: {ms + ms_why:.0f} ms (search {ms:.0f} ms, explanations "
               f"{ms_why:.0f} ms) | job skills recognised: "
               + (", ".join(s["skill"] for s in job_sk.values()) or "none"))


def candidates_for_job(jid, wid, k=K):
    return index.top_users(jid, wid, k=k)


def candidates_for_text(title, requirements, state, window=None, k=K):
    """A job typed by the user (it may have no applications at all): content search over users."""
    state = state.strip().upper()
    cells = {w: len(ix) for (w, s), ix in index.user_cells.items() if s == state}
    if not cells:
        return None, None, None
    if window is None:
        window = max(cells, key=cells.get)
    ix, uidx = index._index(window, state)
    text = f"{title.strip()}. {requirements.strip()}".lower()
    q = rec.model.encode([text], normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")
    scores, nn = ix.search(q, min(k, len(uidx)))
    top = pd.DataFrame({"UserID": index.users["UserID"].values[uidx[nn[0]]],
                        "WindowID": window,
                        "rank": np.arange(1, len(nn[0]) + 1),
                        "score": scores[0]})
    return top, window, text


st.title("TalentMatch")
st.caption("Open jobs of the test period (2012-06-18 onward). A typed profile is a new user with no "
           "application history, so the collaborative signal is empty. Ranking: the LightGBM re-ranker, "
           "or Profile match (content similarity on profile and skills) for cold-start users.")

tab1, tab2 = st.tabs(["Job seeker", "Recruiter"])

with tab1:
    c1, c2, c3 = st.columns(3)
    degree = c1.selectbox("Degree", DEGREES, index=4)
    major = c2.text_input("Major", "Business Administration")
    years = c3.number_input("Years of experience", min_value=0, max_value=40, value=5)
    roles_text = st.text_area("Past roles (one per line)",
                              "Customer Service Representative\nSales Associate")
    skills_text = st.text_area("Skills / CV text (optional, used for 'Why this job?' and Profile match)",
                               "Customer service, SQL, communication")
    c4, c5 = st.columns(2)
    state = c4.selectbox("State", states, index=states.index("CA") if "CA" in states else 0)
    city = c5.text_input("City (optional)", "Los Angeles")
    ranking = st.radio(
        "Ranking", ["Re-ranker (LightGBM v2)", "Profile match (cold start)"], horizontal=True,
        help="Re-ranker: the frozen model; for a new user it leans on city and job popularity. "
             "Profile match: jobs most similar to your degree, major, roles and skills (content only).")

    if st.button("Recommend jobs"):
        roles = [r for r in roles_text.splitlines() if r.strip()]
        deg = "" if degree == DEGREES[0] else degree
        rmode = "profile" if ranking.startswith("Profile") else "reranker"
        t = time.time()
        res = rec.recommend(deg, major, years, roles, state, city, k=K,
                            mode=rmode, skills=skills_text)
        t_rec = (time.time() - t) * 1000

        t = time.time()
        cv_text = ". ".join([skills_text, ". ".join(roles), major])
        cv, expl = why.explain(cv_text, list(zip(res.JobID, res.WindowID)))
        t_why = (time.time() - t) * 1000

        out = res.copy()
        out["score"] = out["score"].round(3)
        out["City"] = out["City"].str.title()
        out["Why"] = [", ".join(e["skills"]) if e["skills"] else "-" for e in expl]
        st.dataframe(out[["rank", "Title", "City", "State", "score", "Why", "JobID", "WindowID"]],
                     width="stretch", hide_index=True)
        st.caption(f"response time: {t_rec + t_why:.0f} ms (ranking {t_rec:.0f} ms, explanations "
                   f"{t_why:.0f} ms) | mode {res.attrs['mode']} | window {res.attrs['window']} | "
                   f"query text: {res.attrs['text']}")
        st.caption("Skills recognised in your text: "
                   + (", ".join(s["skill"] for s in cv.values()) or "none"))
        for rk, title, e in zip(out["rank"], out["Title"], expl):
            if e["evidence"]:
                with st.expander(f"#{rk} {title}: why this job?"):
                    for ev in e["evidence"]:
                        st.write(f"**{ev['skill']}** ({ev['source']}): ...{ev['context']}...")

with tab2:
    st.caption("Candidates are ranked by content similarity (bge-small cosine) between the job "
               "and each user profile, within one window and state. Content only: no "
               "collaborative signal, and the LightGBM re-ranker is not applied in this direction. "
               "Why: ESCO skills of the job that also appear in the candidate's major or past roles.")
    mode = st.radio("Job source", ["New job (type it)", "Existing job (by ID)"], horizontal=True)

    if mode == "New job (type it)":
        title = st.text_input("Job title", "Customer Service Representative")
        reqs = st.text_area("Requirements / description",
                            "Answer inbound customer calls, resolve billing questions and "
                            "document each case. Previous call center experience preferred.")
        jstate = st.selectbox("State of the job", states,
                              index=states.index("CA") if "CA" in states else 0, key="jstate")
        if st.button("Find candidates", key="find_new"):
            t = time.time()
            top, window, text = candidates_for_text(title, reqs, jstate)
            ms = (time.time() - t) * 1000
            if top is None:
                st.warning("No users for this state in the index.")
            else:
                st.write(f"{title} | {jstate}")
                show_candidates(top, ms, typed_job_skills(title, reqs))
                st.caption(f"window {window} | query text: {text}")
    else:
        top_job = rec.jobs.sort_values("job_pop").iloc[-1]
        c1, c2 = st.columns(2)
        jid = c1.number_input("JobID", min_value=0, value=int(top_job.JobID), step=1)
        wid = c2.number_input("WindowID", min_value=0, value=int(top_job.WindowID), step=1)
        if st.button("Find candidates", key="find_old"):
            t = time.time()
            top = candidates_for_job(int(jid), int(wid))
            ms = (time.time() - t) * 1000
            if top.empty:
                st.warning("No candidates for this job (is it in the open-jobs index?).")
            else:
                st.write(job_line(int(jid), int(wid)))
                show_candidates(top, ms, why.job_skills(int(jid), int(wid)))