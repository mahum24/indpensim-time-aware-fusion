"""Repeat-level summary and paired tests. Usage: python -I code/08_paired.py <predictions.csv>
R2 / MAE% / coverage as mean ± sd over repeats; per-batch absolute error averaged over repeats, then Wilcoxon signed-rank
(two-sided) time vs grid, dense and ridge at each horizon, Holm-corrected over all comparisons."""
import sys
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv(sys.argv[1]); df["ae"] = (df.pred - df.y).abs()
def r2(g): return 1 - ((g.pred - g.y) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum()
per = df.groupby(["variant", "k", "rep"]).apply(lambda g: pd.Series({"r2": r2(g), "mae_pct": 100 * g.ae.mean() / g.y.mean(),
    "cover80": (g.ae <= 1.2816 * g["std"]).mean() if g["std"].notna().all() else np.nan,
    "cover95": (g.ae <= 1.96 * g["std"]).mean() if g["std"].notna().all() else np.nan}), include_groups=False)
agg = per.groupby(["variant", "k"]).agg(["mean", "std"])
for m in ["r2", "mae_pct", "cover80", "cover95"]:
    t = agg[m].apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1).unstack("variant")
    print(f"== {m} (mean ± sd over {df.rep.nunique()} repeats)"); print(t.to_string())
e = df.groupby(["variant", "k", "batch"]).ae.mean().unstack("variant")
rows = []
for k, g in e.groupby(level="k"):
    for other in ["grid", "dense", "ridge"]:
        d = g["time"] - g[other]
        p = stats.wilcoxon(g["time"], g[other]).pvalue
        rows.append(dict(k=k, vs=other, median_diff_gL=d.median(), time_better=f"{(d < 0).sum()}/{len(d)}", p=p))
r = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
m = len(r); r["p_holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * r.p))
r = r.sort_values(["k", "vs"])
print("== paired Wilcoxon on per-batch |error| (time minus other; negative = time better), Holm-corrected")
print(r.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
