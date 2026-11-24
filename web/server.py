# web/server.py  (FastAPI backend: same model and data as app/app.py, no model changes)
# Run from the repo root:  python -m uvicorn web.server:app --port 8000
import html
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from profile_recommender import ProfileRecommender  # noqa: E402
from recruiter_candidates import RecruiterIndex  # noqa: E402
from why_skills import WhyMatcher, similar_role  # noqa: E402
import extract_esco_skills_candidate as esco  # noqa: E402

DEGREES = ["(not specified)", "High School", "Vocational", "Associate's",
           "Bachelor's", "Master's", "PhD"]
K = 10
LOCK = threading.Lock()   # one request at a time: FAISS / LightGBM / matcher caches are shared

print("Loading model and data (once)...")
rec = ProfileRecommender()
index = RecruiterIndex()
why = WhyMatcher()
users_all = pd.read_csv("data/processed/users_clean.tsv", sep="\t",
                        usecols=["UserID", "WindowID", "DegreeType", "Major", "TotalYearsExperience"])
_h = pd.read_csv("data/processed/user_history_clean.tsv", sep="\t",
                 usecols=["UserID", "WindowID", "Sequence", "JobTitle"], dtype={"JobTitle": str})
_h = _h.dropna(subset=["JobTitle"])
history = (_h[_h["JobTitle"].str.strip() != ""]
           .sort_values(["UserID", "WindowID", "Sequence"]).reset_index(drop=True))
states = sorted(rec.jobs.State.dropna().unique().tolist())

picks = pd.Series([max(c, key=c.get) for c in (rec.windows_for(s) for s in states) if c])
for w in picks.value_counts().index[:2]:
    why._window(int(w))
rec.model.encode(["warm up"], normalize_embeddings=True)
print("Ready.")

app = FastAPI(title="TalentMatch")


def clean(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    return v


def reasons(skills, role, same_city=False):
    parts = []
    if skills:
        parts.append(", ".join(skills))
    if role:
        parts.append(f"role: {role}")
    if same_city:
        parts.append("same city")
    return " | ".join(parts) if parts else "-"


def past_roles(top, n=5):
    h = history[history["UserID"].isin(top["UserID"])]
    h = h.merge(top[["UserID", "WindowID"]], on=["UserID", "WindowID"])
    roles = (h.groupby(["UserID", "WindowID"])["JobTitle"]
             .apply(lambda s: " ; ".join(s.str.strip().head(n))).rename("Past roles").reset_index())
    return top.merge(roles, on=["UserID", "WindowID"], how="left")


def typed_job_skills(title, requirements):
    t = esco.normalize(title)
    found = esco.match_skills(f"{title}. {requirements}", t, why.matcher, why.root)
    return {s["esco_uri"]: s for s in found}


def recruiter_why(top, job_sk, job_title):
    out = []
    for major, roles in zip(top["Major"], top["Past roles"]):
        text = ". ".join(x for x in (major, roles) if isinstance(x, str) and x.strip())
        cand = why.cv_skills(text)
        shared = [job_sk[u]["skill"] for u in cand if u in job_sk]
        role_list = roles.split(" ; ") if isinstance(roles, str) else []
        out.append({"skills": shared, "role": similar_role(job_title, role_list),
                    "text": reasons(shared, similar_role(job_title, role_list))})
    return out


def candidates_payload(top, ms, job_sk, job_title, header, window=None, text=None):
    t = time.time()
    top = top.merge(users_all, on=["UserID", "WindowID"], how="left")
    top["Major"] = top["Major"].map(lambda m: html.unescape(m) if isinstance(m, str) else m)
    top = past_roles(top).sort_values("rank")
    whys = recruiter_why(top, job_sk, job_title)
    ms_why = (time.time() - t) * 1000
    rows = []
    for (_, r), w in zip(top.iterrows(), whys):
        rows.append({"rank": clean(r["rank"]), "user_id": clean(r["UserID"]),
                     "degree": clean(r.get("DegreeType")), "major": clean(r.get("Major")),
                     "years": clean(r.get("TotalYearsExperience")),
                     "past_roles": clean(r.get("Past roles")),
                     "score": round(float(r["score"]), 3), "why": w})
    return {"header": header, "rows": rows, "window": clean(window), "query_text": text,
            "ms_search": round(ms), "ms_why": round(ms_why), "ms_total": round(ms + ms_why),
            "job_skills": [s["skill"] for s in job_sk.values()]}


class ProfileIn(BaseModel):
    degree: str = "Bachelor's"
    major: str = ""
    years: int = 0
    roles: str = ""
    skills: str = ""
    state: str
    city: str = ""
    mode: str = "reranker"      # "reranker" | "profile"


class JobIn(BaseModel):
    title: str
    requirements: str = ""
    state: str


class JobIdIn(BaseModel):
    job_id: int
    window_id: int


@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/api/meta")
def meta():
    top_job = rec.jobs.sort_values("job_pop").iloc[-1]
    return {"states": states, "degrees": DEGREES,
            "example_job": {"job_id": int(top_job.JobID), "window_id": int(top_job.WindowID)}}


@app.post("/api/jobs")
def jobs_for_profile(p: ProfileIn):
    roles = [r.strip() for r in p.roles.splitlines() if r.strip()]
    deg = "" if p.degree == DEGREES[0] else p.degree
    mode = "profile" if p.mode == "profile" else "reranker"
    with LOCK:
        t = time.time()
        res = rec.recommend(deg, p.major, p.years, roles, p.state, p.city, k=K,
                            mode=mode, skills=p.skills)
        t_rec = (time.time() - t) * 1000
        t = time.time()
        cv_text = ". ".join([p.skills, ". ".join(roles), p.major])
        cv, expl = why.explain(cv_text, list(zip(res.JobID, res.WindowID)))
        t_why = (time.time() - t) * 1000
    typed_city = p.city.strip().upper()
    rows = []
    for (_, r), e in zip(res.iterrows(), expl):
        role = similar_role(r["Title"], roles)
        same_city = bool(typed_city) and r["City"] == typed_city
        rows.append({
            "rank": clean(r["rank"]), "title": r["Title"], "city": str(r["City"]).title(),
            "state": r["State"], "score": round(float(r["score"]), 3),
            "job_id": clean(r["JobID"]), "window_id": clean(r["WindowID"]),
            "why": {"skills": e["skills"], "role": role, "same_city": same_city,
                    "text": reasons(e["skills"], role, same_city)},
            "evidence": [{"skill": ev["skill"], "source": ev["source"], "context": ev["context"]}
                         for ev in e["evidence"]]})
    return {"rows": rows, "mode": res.attrs.get("mode"), "window": clean(res.attrs.get("window")),
            "query_text": res.attrs.get("text"),
            "ms_rank": round(t_rec), "ms_why": round(t_why), "ms_total": round(t_rec + t_why),
            "cv_skills": [s["skill"] for s in cv.values()]}


@app.post("/api/candidates")
def candidates_new_job(j: JobIn):
    state = j.state.strip().upper()
    with LOCK:
        t = time.time()
        cells = {w: len(ix) for (w, s), ix in index.user_cells.items() if s == state}
        if not cells:
            raise HTTPException(404, "No users for this state in the index.")
        window = max(cells, key=cells.get)
        ix, uidx = index._index(window, state)
        text = f"{j.title.strip()}. {j.requirements.strip()}".lower()
        q = rec.model.encode([text], normalize_embeddings=True,
                             convert_to_numpy=True).astype("float32")
        scores, nn = ix.search(q, min(K, len(uidx)))
        top = pd.DataFrame({"UserID": index.users["UserID"].values[uidx[nn[0]]],
                            "WindowID": window, "rank": np.arange(1, len(nn[0]) + 1),
                            "score": scores[0]})
        ms = (time.time() - t) * 1000
        return candidates_payload(top, ms, typed_job_skills(j.title, j.requirements), j.title,
                                  f"{j.title} | {state}", window, text)


@app.post("/api/candidates_by_id")
def candidates_by_id(j: JobIdIn):
    with LOCK:
        t = time.time()
        top = index.top_users(j.job_id, j.window_id, k=K)
        ms = (time.time() - t) * 1000
        if top.empty:
            raise HTTPException(404, "No candidates for this job (is it in the open-jobs index?).")
        m = rec.jobs[(rec.jobs.JobID == j.job_id) & (rec.jobs.WindowID == j.window_id)]
        title = "" if m.empty else str(m.iloc[0].Title)
        header = "Job not found" if m.empty else f"{title} | {str(m.iloc[0].City).title()}, {m.iloc[0].State}"
        return candidates_payload(top, ms, why.job_skills(j.job_id, j.window_id), title, header,
                                  j.window_id)
