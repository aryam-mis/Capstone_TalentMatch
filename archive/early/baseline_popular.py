import pandas as pd

K = 150
tr = pd.read_csv("results/split/train.csv", usecols=["UserID", "WindowID", "JobID"])
val = pd.read_csv("results/split/val.csv", usecols=["UserID"]).drop_duplicates()

top = (tr.groupby(["JobID", "WindowID"]).size()
         .sort_values(ascending=False).head(K).reset_index()[["JobID", "WindowID"]])
top["rank"] = range(1, len(top) + 1)

recs = val.merge(top, how="cross")
recs.to_csv("results/recs_popular_global.csv", index=False)
print("users:", val.UserID.nunique(), "| recs per user:", K, "| rows:", len(recs))
