"""Quick facts about the converted IndPenSim data. Usage: python -I code/02_explore.py <data_dir> <stats_csv>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
d = Path(sys.argv[1]); p = pd.read_parquet(d / "process.parquet"); st = pd.read_csv(sys.argv[2])
c = list(p.columns)
print("process columns:", [x.split("(")[0] for x in c[:31]])
g = p.groupby("Batch")
dt = g["Time (h)"].apply(lambda s: np.median(np.diff(s.to_numpy())))
print("rows/batch min/med/max:", g.size().min(), g.size().median(), g.size().max(), "| duration h min/max:", g["Time (h)"].max().min(), g["Time (h)"].max().max(), "| median dt h:", dt.median())
off = [x for x in c if "offline" in x.lower()]
for x in off:
    cnt = g[x].apply(lambda s: s.notna().sum())
    gaps = g.apply(lambda df: np.median(np.diff(df.loc[df[x].notna(), "Time (h)"].to_numpy())) if df[x].notna().sum() > 1 else np.nan, include_groups=False)
    print(f"offline {x.split('(')[0]!r}: samples/batch med {cnt.median():.0f}, median gap {np.nanmedian(gaps):.1f} h")
ff = p[p.Fault_flag > 0].groupby("Batch")["Time (h)"].agg(["min", "max", "count"])
print("fault onset per faulty batch (h):\n", ff.to_string())
for b in [1, 61, 91]:
    r = np.load(d / f"raman_{b:03d}.npy")
    nz = (np.abs(r).sum(1) > 0)
    t = p.loc[p.Batch == b, "Time (h)"].to_numpy()
    print(f"batch {b}: raman {r.shape}, spectra from {t[np.abs(np.nan_to_num(r)).sum(1)>0][0]} h")
st.columns = ["batch", "harv_during", "harv_end", "yield_total", "fault"]
print("yield (kg) by group:")
grp = pd.cut(st.batch, [0, 30, 60, 90, 100], labels=["recipe 1-30", "operator 31-60", "Raman/APC 61-90", "faults 91-100"])
print(st.groupby(grp, observed=True)["yield_total"].describe()[["mean", "std", "min", "max"]].round(-3).to_string())
