"""Does predictive uncertainty drop when a sparse assay arrives? Usage: python -I code/07_uncertainty.py <hourly.npy> <traces.csv>
For each test batch and hour k (24..144), compares the hour-to-hour change in predicted std (k-1 -> k) at hours where a new
offline assay enters the model (assay time in (k-1, k]) with hours where only one more hour of dense data enters."""
import sys
import numpy as np, pandas as pd
from scipy import stats
D = np.load(sys.argv[1], allow_pickle=True).item(); B = D["batches"]
tr = pd.read_csv(sys.argv[2]).sort_values(["rep", "batch", "k"])
rows = []
for (rep, b), g in tr.groupby(["rep", "batch"]):
    g = g.set_index("k"); ot = B[int(b)]["off_t"]
    for k in range(25, 145):
        if k - 1 not in g.index or k not in g.index: continue
        s0, s1 = g.at[k - 1, "std"], g.at[k, "std"]; e0, e1 = abs(g.at[k - 1, "pred"] - g.at[k - 1, "y"]), abs(g.at[k, "pred"] - g.at[k, "y"])
        assay = bool(((ot > k - 1) & (ot <= k)).any())
        rows.append(dict(batch=b, k=k, assay=assay, dstd_pct=100 * (s1 - s0) / s0, derr=e1 - e0))
r = pd.DataFrame(rows)
a, n = r[r.assay], r[~r.assay]
print(f"assay hours: {len(a)}, other hours: {len(n)}")
print(f"mean change in predicted std: assay hours {a.dstd_pct.mean():+.2f}%  vs other hours {n.dstd_pct.mean():+.2f}%")
print(f"median change: assay {a.dstd_pct.median():+.2f}%  vs other {n.dstd_pct.median():+.2f}%")
print(f"share of steps where std fell: assay {100 * (a.dstd_pct < 0).mean():.0f}%  vs other {100 * (n.dstd_pct < 0).mean():.0f}%")
print("Mann-Whitney (assay < other) p =", f"{stats.mannwhitneyu(a.dstd_pct, n.dstd_pct, alternative='less').pvalue:.2e}")
print(f"mean change in |error| (g/L): assay {a.derr.mean():+.3f} vs other {n.derr.mean():+.3f}")
t2 = tr[tr.k.isin([24, 48, 72, 96, 120, 144])].copy(); t2["err"] = (t2.pred - t2.y).abs()
print("mean predicted std vs mean |error| by horizon (g/L):")
print(t2.groupby("k")[["std", "err"]].mean().round(2).to_string())
