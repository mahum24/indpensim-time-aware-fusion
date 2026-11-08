"""Is a FIXED-TIME quality target predictable early? Target: offline penicillin concentration (g/L) at the assay nearest 160 h
(every batch lasts >= 167 h). Same features and CV as 04, but the "last" value is the mean of the last 6 h (single-hour spikes in batches 42, 67, 91 at 96 h broke the last-hour version). Usage: python -I code/05_fixed_target.py <hourly.npy>"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
D = np.load(sys.argv[1], allow_pickle=True).item(); B = D["batches"]; ids = sorted(B); off = D["offline"]
pi = [i for i, c in enumerate(off) if c.startswith("Offline Penicillin")][0]
def target(b):
    t = B[b]["off_t"]; j = np.argmin(np.abs(t - 160)); return B[b]["off_v"][j, pi], t[j]
yt = np.array([target(b) for b in ids]); y = yt[:, 0]; grp = np.array([B[b]["group"] for b in ids])
print("target time range", yt[:, 1].min(), yt[:, 1].max(), "| P@160h mean", y.mean().round(2), "sd", y.std().round(2))
def feats(b, k):
    x = B[b]["proc"][:k]; x = np.nan_to_num(x, nan=np.nanmean(x, 0)); t = np.arange(len(x))
    m = B[b]["off_t"] <= k
    return np.concatenate([x.mean(0), x[-6:].mean(0), np.polyfit(t, x, 1)[0], B[b]["off_v"][m][-1] if m.any() else np.zeros(5)])
for k in [24, 48, 72, 96, 120]:
    X = np.array([feats(b, k) for b in ids]); R = np.array([np.nanmean(B[b]["raman"][k - 6:k], 0) for b in ids])
    X = np.clip(X, *np.percentile(X, [1, 99], axis=0))
    out = []
    for rep in range(3):
        pr = np.zeros(len(y))
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=rep).split(X, grp):
            pca = make_pipeline(StandardScaler(), PCA(10, random_state=0)).fit(R[tr])
            a, c = np.c_[X[tr], pca.transform(R[tr])], np.c_[X[te], pca.transform(R[te])]
            pr[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(a, y[tr]).predict(c)
        out.append((100 * np.abs(pr - y).mean() / y.mean(), 1 - ((pr - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    o = np.array(out); print(f"k={k:3d}h  MAE {o[:,0].mean():.1f}%  R2 {o[:,1].mean():.3f}")
print("naive MAE %", round(100 * np.abs(y - y.mean()).mean() / y.mean(), 1))
