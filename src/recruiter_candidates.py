import argparse
import os

import numpy as np
import pandas as pd
import faiss

JOBS_PARQUET = "artifacts/jobs_open_union.parquet"
JOBS_EMB = "artifacts/job_union_bge_small_title_req.npy"
USERS_KEYS = "artifacts/users_keys.parquet"
USERS_EMB = "artifacts/user_emb_bge_small.npy"
KEY = ["UserID", "WindowID", "JobID"]


def load_assets():
    jobs = pd.read_parquet(JOBS_PARQUET).reset_index(drop=True)
    J = np.load(JOBS_EMB, mmap_mode="r")
    users = pd.read_parquet(USERS_KEYS).reset_index(drop=True)
    U = np.load(USERS_EMB, mmap_mode="r")
    assert len(J) == len(jobs), (len(J), len(jobs))
    assert len(U) == len(users), (len(U), len(users))
    return jobs, J, users, U


def open_mask(jobs, start, end):
    m = ((jobs["EndDate"].isna() | (jobs["EndDate"] >= start)) &
         (jobs["StartDate"].isna() | (jobs["StartDate"] < end)) &
         ~jobs.duplicated(["JobID", "WindowID"]))
    return m.values


def generate(jobs, J, users, U, mask, k_retrieve):
    """For every open job, retrieve the nearest users inside the same (WindowID, State) cell."""
    user_cells = users.groupby(["WindowID", "State"]).indices
    job_rows = np.flatnonzero(mask)
    sub = jobs.iloc[job_rows]
    job_ids = jobs["JobID"].values
    job_win = jobs["WindowID"].values
    user_ids = users["UserID"].values

    parts = []
    for (w, st), pos in sub.groupby(["WindowID", "State"]).indices.items():
        uidx = user_cells.get((w, st))
        if uidx is None:
            continue
        jpos = job_rows[pos]
        index = faiss.IndexFlatIP(U.shape[1])
        index.add(np.ascontiguousarray(U[uidx], dtype="float32"))
        kk = min(k_retrieve, len(uidx))
        scores, nn = index.search(np.ascontiguousarray(J[jpos], dtype="float32"), kk)
        parts.append(pd.DataFrame({
            "JobID": np.repeat(job_ids[jpos], kk),
            "WindowID": np.repeat(job_win[jpos], kk),
            "UserID": user_ids[uidx[nn]].ravel(),
            "score": scores.ravel().astype("float32"),
            "rank": np.tile(np.arange(1, kk + 1, dtype=np.int32), len(jpos)),
        }))
    return pd.concat(parts, ignore_index=True)


def finalize(recs, k):
    recs = recs.sort_values(["JobID", "WindowID", "rank"], kind="stable")
    recs = recs.drop_duplicates(["JobID", "WindowID", "UserID"])
    recs["rank"] = (recs.groupby(["JobID", "WindowID"]).cumcount() + 1).astype("int32")
    return recs[recs["rank"] <= k][["JobID", "WindowID", "UserID", "rank", "score"]]


def evaluate(recs, labels, jobs, users, ks):
    """Recall of the real applicants among the top-k users, plus a random-ranking reference."""
    jobs_u = (jobs[["JobID", "WindowID", "State"]].drop_duplicates(["JobID", "WindowID"])
              .rename(columns={"State": "job_state"}))
    users_u = (users[["UserID", "WindowID", "State"]].drop_duplicates(["UserID", "WindowID"])
               .rename(columns={"State": "user_state"}))
    cell = (users.groupby(["WindowID", "State"]).size().rename("cell_size").reset_index()
            .rename(columns={"State": "job_state"}))

    lab = labels[KEY].drop_duplicates()
    lab = lab.merge(users_u, on=["UserID", "WindowID"], how="left").merge(
        jobs_u, on=["JobID", "WindowID"], how="left")
    lab["in_cell"] = (lab["user_state"] == lab["job_state"]).astype(float)

    per_job = (lab.groupby(["JobID", "WindowID"])
               .agg(n_app=("UserID", "size"), in_cell=("in_cell", "mean")).reset_index())
    rec_jobs = recs[["JobID", "WindowID"]].drop_duplicates()
    per_job = (per_job.merge(rec_jobs, on=["JobID", "WindowID"])
               .merge(jobs_u, on=["JobID", "WindowID"], how="left")
               .merge(cell, on=["WindowID", "job_state"], how="left"))
    print("jobs evaluated (open jobs with at least one real applicant):", len(per_job))
    print("mean real applicants per job:", round(per_job["n_app"].mean(), 2))
    print("share of applicants inside the candidate pool (same window and state):",
          round(per_job["in_cell"].mean(), 4))

    hit = recs.merge(lab[KEY].assign(hit=1), on=["JobID", "WindowID", "UserID"], how="left")
    hit["hit"] = hit["hit"].fillna(0)
    for k in ks:
        h = (hit[hit["rank"] <= k].groupby(["JobID", "WindowID"])["hit"].sum()
             .rename("hits").reset_index())
        d = per_job.merge(h, on=["JobID", "WindowID"], how="left").fillna({"hits": 0})
        recall = (d["hits"] / d["n_app"]).mean()
        random_ref = (d["in_cell"] * np.minimum(k, d["cell_size"]) / d["cell_size"]).mean()
        print(f"Recall@{k}: {recall:.4f} | random ranking inside the pool: {random_ref:.4f}")


class RecruiterIndex:
    """Lookup used by the app: top users for one job, or the score of one (user, job) pair."""

    def __init__(self):
        self.jobs, self.J, self.users, self.U = load_assets()
        self.job_pos = {(j, w): i for i, (j, w) in
                        enumerate(zip(self.jobs["JobID"], self.jobs["WindowID"]))}
        self.user_pos = {(u, w): i for i, (u, w) in
                         enumerate(zip(self.users["UserID"], self.users["WindowID"]))}
        self.user_cells = self.users.groupby(["WindowID", "State"]).indices
        self._cache = {}

    def _index(self, window_id, state):
        key = (window_id, state)
        if key not in self._cache:
            uidx = self.user_cells.get(key)
            if uidx is None:
                self._cache[key] = None
            else:
                ix = faiss.IndexFlatIP(self.U.shape[1])
                ix.add(np.ascontiguousarray(self.U[uidx], dtype="float32"))
                self._cache[key] = (ix, uidx)
        return self._cache[key]

    def top_users(self, job_id, window_id, k=10):
        empty = pd.DataFrame(columns=["UserID", "WindowID", "rank", "score"])
        i = self.job_pos.get((job_id, window_id))
        if i is None:
            return empty
        entry = self._index(window_id, self.jobs["State"].values[i])
        if entry is None:
            return empty
        ix, uidx = entry
        q = np.ascontiguousarray(self.J[i:i + 1], dtype="float32")
        scores, nn = ix.search(q, min(k, len(uidx)))
        return pd.DataFrame({"UserID": self.users["UserID"].values[uidx[nn[0]]],
                             "WindowID": window_id,
                             "rank": np.arange(1, len(nn[0]) + 1),
                             "score": scores[0]})

    def score_pair(self, user_id, window_id, job_id):
        ui = self.user_pos.get((user_id, window_id))
        ji = self.job_pos.get((job_id, window_id))
        if ui is None or ji is None:
            return None
        return float(np.dot(self.U[ui].astype("float32"), self.J[ji].astype("float32")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--period_start", required=True)
    p.add_argument("--period_end", required=True)
    p.add_argument("--history_before", default=None,
                   help="exclude users who already applied to the job before this date")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--name", default="recruiter")
    p.add_argument("--sample", type=int, default=0, help="random sample of jobs, for quick tests")
    p.add_argument("--eval_labels", default=None, help="e.g. results/recs/val_labels.csv")
    args = p.parse_args()

    os.makedirs("results/recs", exist_ok=True)
    start, end = pd.Timestamp(args.period_start), pd.Timestamp(args.period_end)
    jobs, J, users, U = load_assets()
    mask = open_mask(jobs, start, end)
    print("open jobs in period:", int(mask.sum()))

    if args.sample:
        rng = np.random.default_rng(42)
        chosen = rng.choice(np.flatnonzero(mask), size=min(args.sample, int(mask.sum())), replace=False)
        mask = np.zeros(len(jobs), dtype=bool)
        mask[chosen] = True
        print("sampled jobs:", int(mask.sum()))

    recs = generate(jobs, J, users, U, mask, int(args.k * 1.5) + 5)
    print("raw rows:", len(recs))

    if args.history_before:
        from split_utils import load_apps
        apps = load_apps()
        seen = (apps[apps["ApplicationDate"] < pd.Timestamp(args.history_before)][KEY]
                .drop_duplicates().assign(_seen=1))
        recs = recs.merge(seen, on=KEY, how="left")
        recs = recs[recs["_seen"].isna()].drop(columns="_seen")

    recs = finalize(recs, args.k)
    out = f"results/recs/{args.name}.csv"
    recs.to_csv(out, index=False)
    print("saved", out, len(recs), "| jobs with candidates:",
          recs[["JobID", "WindowID"]].drop_duplicates().shape[0])

    if args.eval_labels:
        labels = pd.read_csv(args.eval_labels)
        evaluate(recs, labels, jobs, users, ks=sorted({10, args.k}))


if __name__ == "__main__":
    main()