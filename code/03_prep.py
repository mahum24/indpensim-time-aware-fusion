"""Hourly arrays per batch for modelling. Usage: python -I code/03_prep.py <data_dir> <stats_csv> <out_npz>
Per batch: process (T,P) hourly means of online variables, raman (T,2200) hourly mean spectra (NaN before 3 h),
offline list of (time_h, 5 values), target = total penicillin yield, fault label, control group."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
d = Path(sys.argv[1]); p = pd.read_parquet(d / "process.parquet"); st = pd.read_csv(sys.argv[2])
cols = list(p.columns)
offline = [c for c in cols[:31] if "offline" in c.lower() or "off-line" in c.lower() or c.startswith("Viscosity")]
# drop offline cols and the simulator's own (noise-free) penicillin/biomass states that a plant would not measure online
online = [c for c in cols[1:31] if c not in offline and not c.startswith("Penicillin concentration")]
print("online vars:", len(online), "| offline vars:", [c.split("(")[0] for c in offline])
out = {}
for b, df in p.groupby("Batch"):
    b = int(b); t = df["Time (h)"].to_numpy(); hr = np.floor(t - 1e-6).astype(int)
    T = hr.max() + 1
    proc = pd.DataFrame(df[online].to_numpy(), index=hr).groupby(level=0).mean().reindex(range(T)).to_numpy(np.float32)
    ram = np.load(d / f"raman_{b:03d}.npy"); ok = np.abs(np.nan_to_num(ram)).sum(1) > 0
    ram_h = pd.DataFrame(np.where(ok[:, None], ram, np.nan), index=hr).groupby(level=0).mean().reindex(range(T)).to_numpy(np.float32)
    om = df[offline].notna().all(1).to_numpy()
    off_t = t[om].astype(np.float32); off_v = df.loc[om, offline].to_numpy(np.float32)
    s = st.iloc[b - 1]
    out[b] = dict(proc=proc, raman=ram_h, off_t=off_t, off_v=off_v, y=float(s.iloc[3]), fault=int(s.iloc[4]),
                  group=0 if b <= 30 else 1 if b <= 60 else 2 if b <= 90 else 3, T=T)
np.save(sys.argv[3], {"batches": out, "online": online, "offline": offline}, allow_pickle=True)
Ts = [v["T"] for v in out.values()]; no = [len(v["off_t"]) for v in out.values()]
print("batches", len(out), "hours min/med/max", min(Ts), int(np.median(Ts)), max(Ts), "| offline samples/batch", min(no), int(np.median(no)), max(no))
