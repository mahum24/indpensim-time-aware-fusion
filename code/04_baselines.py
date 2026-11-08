"""Baseline: predict final penicillin yield from the first k hours. Usage: python -I code/04_baselines.py <hourly.npy> <out_csv>
Features at horizon k: per online variable mean/last/slope over [0,k]; Raman PCA (fit inside each training fold) of the
mean spectrum over the last 6 h before k; last offline assay values before k. 10-fold CV x 3 repeats, grouped by batch."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

D = np.load(sys.argv[1], allow_pickle=True).item(); B = D["batches"]; ids = sorted(B)
y = np.array([B[b]["y"] for b in ids]) / 1e6; grp = np.array([B[b]["group"] for b in ids])

def proc_feats(b, k):
    x = B[b]["proc"][:k]; x = np.nan_to_num(x, nan=np.nanmean(x, 0))
    t = np.arange(len(x)); slope = np.polyfit(t, x, 1)[0] if len(x) > 1 else np.zeros(x.shape[1])
    return np.concatenate([x.mean(0), x[-1], slope])
def raman_last(b, k):
    r = B[b]["raman"][max(0, k - 6):k]; return np.nanmean(r, 0)
def off_feats(b, k):
    m = B[b]["off_t"] <= k
    return np.concatenate([B[b]["off_v"][m][-1] if m.any() else np.zeros(5), [m.sum()]])

rows = []
for k in [24, 48, 72, 96, 120]:
    P = np.array([proc_feats(b, k) for b in ids]); R = np.array([raman_last(b, k) for b in ids]); O = np.array([off_feats(b, k) for b in ids])
    for rep in range(3):
        cv = StratifiedKFold(10, shuffle=True, random_state=rep)
        preds = {n: np.zeros(len(ids)) for n in ["process", "process+offline", "process+raman", "all (ridge)", "raman PLS"]}
        for tr, te in cv.split(P, grp):
            pca = make_pipeline(StandardScaler(), PCA(10, random_state=0)).fit(R[tr]); Rtr, Rte = pca.transform(R[tr]), pca.transform(R[te])
            sets = {"process": (P[tr], P[te]), "process+offline": (np.c_[P[tr], O[tr]], np.c_[P[te], O[te]]),
                    "process+raman": (np.c_[P[tr], Rtr], np.c_[P[te], Rte]), "all (ridge)": (np.c_[P[tr], O[tr], Rtr], np.c_[P[te], O[te], Rte])}
            for n, (a, c) in sets.items():
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(a, y[tr]); preds[n][te] = m.predict(c)
            pls = make_pipeline(StandardScaler(), PLSRegression(5)).fit(R[tr], y[tr]); preds["raman PLS"][te] = pls.predict(R[te]).ravel()
        for n, pr in preds.items():
            mae = np.abs(pr - y).mean(); r2 = 1 - ((pr - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            rows.append(dict(k=k, model=n, rep=rep, mae_pct=100 * mae / y.mean(), r2=r2))
res = pd.DataFrame(rows).groupby(["k", "model"])[["mae_pct", "r2"]].agg(["mean", "std"]).round(3)
res.to_csv(sys.argv[2]); print(res.to_string())
print("naive (predict train mean) MAE %:", round(100 * np.abs(y - y.mean()).mean() / y.mean(), 1))
