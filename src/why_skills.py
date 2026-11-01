# src/why_skills.py
# "Why this job?": ESCO skills shared by the CV text and a recommended job, with evidence snippets
import argparse
import re

import pandas as pd

import extract_esco_skills_candidate as esco

JOBS_TEXT = "data/processed/jobs_text_open.parquet"


# words that say nothing about the occupation itself, ignored when comparing job titles
GENERIC = {"senior", "junior", "sr", "jr", "lead", "staff", "associate", "assistant", "specialist",
           "representative", "rep", "entry", "level", "full", "part", "time", "temp", "position",
           "job", "jobs", "needed", "wanted", "opportunity", "hiring", "immediate", "and", "the",
           "for", "with", "of", "in", "at", "to", "a", "an", "i", "ii", "iii", "iv", "pay",
           "competitive", "great", "new", "company", "busy", "downtown", "firm"}


def title_tokens(title):
    words = re.findall(r"[a-z][a-z+#]*", str(title).lower())
    return {w for w in words if len(w) >= 3 and w not in GENERIC}


def similar_role(job_title, roles):
    """The past role that shares the most occupation words with the job title, or None."""
    jt = title_tokens(job_title)
    best, best_n = None, 0
    for r in roles:
        if not isinstance(r, str) or not r.strip():
            continue
        n = len(jt & title_tokens(r))
        if n > best_n:
            best, best_n = r.strip(), n
    return best


def snippet(text, label, width=80):
    m = re.search(r"\b" + re.escape(label) + r"\b", text, re.I)
    if not m:
        return ""
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return " ".join(text[a:b].split())


class WhyMatcher:
    def __init__(self, path=JOBS_TEXT):
        self.matcher, self.root = esco.build_matcher()
        self.path = path
        self._win = {}          # window -> DataFrame indexed by JobID
        self._cache = {}        # (JobID, WindowID) -> {esco_uri: info}

    def _window(self, w):
        w = int(w)
        if w not in self._win:
            d = pd.read_parquet(self.path, filters=[("WindowID", "==", w)])
            self._win[w] = d.drop_duplicates("JobID").set_index("JobID")
        return self._win[w]

    def cv_skills(self, cv_text):
        if not cv_text or not cv_text.strip():
            return {}
        found = esco.match_skills(cv_text, "", self.matcher, self.root)
        return {s["esco_uri"]: s for s in found}

    def job_skills(self, job_id, window):
        key = (int(job_id), int(window))
        if key not in self._cache:
            d = self._window(window)
            info = {}
            if key[0] in d.index:
                row = d.loc[key[0]]
                title = esco.normalize(row["Title"])
                req = esco.remove_footer(row["Requirements"])
                desc, _ = esco.select_description(row["Description"], title)
                for source, text in (("Requirements", req), ("Description", desc)):
                    for s in esco.match_skills(text, title, self.matcher, self.root):
                        if s["esco_uri"] not in info:
                            info[s["esco_uri"]] = {
                                "skill": s["skill"], "matched_phrase": s["matched_label"],
                                "source": source,
                                "context": snippet(text, s["matched_label"])}
            self._cache[key] = info
        return self._cache[key]

    def explain(self, cv_text, jobs):
        """jobs: iterable of (JobID, WindowID). Returns (cv skills, one dict per job)."""
        cv = self.cv_skills(cv_text)
        out = []
        for jid, w in jobs:
            js = self.job_skills(jid, w)
            shared = [js[u] for u in cv if u in js]
            out.append({"skills": [x["skill"] for x in shared], "evidence": shared})
        return cv, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    wm = WhyMatcher()
    jobs = [(j, a.window) for j in wm._window(a.window).index[:a.n]]
    cv, res = wm.explain(a.cv, jobs)
    print("CV skills:", [s["skill"] for s in cv.values()])
    shown = 0
    for (jid, w), r in zip(jobs, res):
        if r["skills"] and shown < 6:
            shown += 1
            print(f"\nJob {jid} (window {w}) -> {r['skills']}")
            for e in r["evidence"][:2]:
                print("  ", e["skill"], "|", e["source"], "|", e["context"])
    print("\njobs with at least one match:", sum(1 for r in res if r["skills"]), "of", len(res))


if __name__ == "__main__":
    main()