import pandas as pd

K = 150

tr = pd.read_csv("results/split/train.csv", usecols=["UserID", "WindowID", "JobID"])
val_users = pd.read_csv("results/split/val.csv", usecols=["UserID"]).drop_duplicates()

jobs = pd.read_csv("data/jobs/jobs.tsv", sep="\t", usecols=["JobID", "WindowID", "State"],
                   on_bad_lines="skip", dtype={"State": str})
users = pd.read_csv("data/users.tsv", sep="\t", usecols=["UserID", "WindowID", "State"],
                    dtype={"State": str})

# شهرة كل وظيفة من train فقط
pop = tr.groupby(["JobID", "WindowID"]).size().rename("n").reset_index()
pop = pop.merge(jobs, on=["JobID", "WindowID"], how="inner")

# أشهر K وظيفة لكل (نافذة، ولاية)
pop = pop.sort_values(["WindowID", "State", "n"], ascending=[True, True, False])
pop["rank"] = pop.groupby(["WindowID", "State"]).cumcount() + 1
pop = pop[pop["rank"] <= K][["WindowID", "State", "JobID", "rank"]]

# ربط المستخدمين بنافذتهم وولايتهم
vu = val_users.merge(users, on="UserID", how="left")
print("val users:", len(vu), "| without state/window info:", vu.State.isna().sum())
vu = vu.dropna(subset=["State"])

recs = vu.merge(pop, on=["WindowID", "State"], how="inner")[["UserID", "WindowID", "JobID", "rank"]]
recs.to_csv("results/recs_popular_state.csv", index=False)
print("users with recs:", recs.UserID.nunique(), "| rows:", len(recs))
