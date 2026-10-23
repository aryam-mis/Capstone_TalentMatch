import pandas as pd
from split_utils import load_apps, temporal_split

train, val, _ = temporal_split(load_apps())
K = ["UserID", "WindowID", "JobID"]

# Validation targets = val applications not already seen in train (same rule as evaluate.py)
seen = train[K].drop_duplicates().assign(_s=1)
truth = val[K].drop_duplicates().merge(seen, on=K, how="left")
truth = truth[truth["_s"].isna()].drop(columns="_s")
print("val target pairs:", len(truth))

# Item side: did the target job get at least one application in train?
train_jobs = set(zip(train.JobID, train.WindowID))
in_train = pd.Series([(j, w) in train_jobs for j, w in zip(truth.JobID, truth.WindowID)])
print("share of val targets with >=1 train application:", round(in_train.mean(), 4))

# User side: do validation users have history in train?
pairs = truth[["UserID", "WindowID"]].drop_duplicates()
train_user_window = set(zip(train.UserID, train.WindowID))
has_hist_w = pd.Series([(u, w) in train_user_window for u, w in zip(pairs.UserID, pairs.WindowID)])
print("share of val (user, window) with train history:", round(has_hist_w.mean(), 4))
print("share of val users (any window) seen in train:", round(pairs.UserID.isin(set(train.UserID)).mean(), 4))