import pandas as pd
import yaml
from pathlib import Path

cfg = yaml.safe_load(open("configs/split.yaml", encoding="utf-8"))
train_end = pd.Timestamp(cfg["train_end"])
val_end = pd.Timestamp(cfg["val_end"])

a = pd.read_csv("data/apps/apps.tsv", sep="\t", dtype={"ApplicationDate": str})
n0 = len(a)
a["ApplicationDate"] = pd.to_datetime(a.ApplicationDate, errors="coerce").dt.normalize()
a = a.dropna(subset=["ApplicationDate"]).drop(columns=["Split"])
print("rows read:", n0, "| dropped bad dates:", n0 - len(a))

out = Path("results/split")
out.mkdir(parents=True, exist_ok=True)

parts = {
    "train": a[a.ApplicationDate <= train_end],
    "val": a[(a.ApplicationDate > train_end) & (a.ApplicationDate <= val_end)],
    "test": a[a.ApplicationDate > val_end],
}
for name, df in parts.items():
    df.to_csv(out / f"{name}.csv", index=False)
    print(name, "rows:", len(df), "| users:", df.UserID.nunique(),
          "| range:", df.ApplicationDate.min().date(), "->", df.ApplicationDate.max().date())

assert sum(len(d) for d in parts.values()) == len(a)
print("OK: no row lost, no overlap")
