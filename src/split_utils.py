import pandas as pd

def load_apps(path="data/apps/apps.tsv"):
    apps = pd.read_csv(path, sep="\t")
    apps["ApplicationDate"] = pd.to_datetime(apps["ApplicationDate"], errors="coerce")
    return apps.dropna(subset=["ApplicationDate"]).reset_index(drop=True)

def temporal_split(apps):
    # Half-open intervals so no row falls between boundaries
    d = apps["ApplicationDate"]
    train = apps[d < "2012-06-06"]
    val   = apps[(d >= "2012-06-06") & (d < "2012-06-14")]
    test  = apps[d >= "2012-06-14"]
    return train, val, test

if __name__ == "__main__":
    train, val, test = temporal_split(load_apps())
    print(len(train), len(val), len(test))
    print("expected:", 1290087, 154298, 153355)