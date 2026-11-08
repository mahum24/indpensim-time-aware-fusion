"""Time-aware cross-attention with evidential (NIG) uncertainty vs grid / dense-only variants and Ridge, on IndPenSim.
Target: offline penicillin (g/L) at 156 h, predicted from data up to horizon k (one model for all horizons).
Usage: python -I code/06_attention.py <hourly.npy> <out_dir> [--folds 10] [--reps 1] [--epochs 400] [--variants time,grid,dense]
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("hourly"); ap.add_argument("out")
ap.add_argument("--folds", type=int, default=10); ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--epochs", type=int, default=400); ap.add_argument("--variants", default="time,grid,dense")
ap.add_argument("--max_folds", type=int, default=99)
ap.add_argument("--d", type=int, default=32); ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--wd", type=float, default=1e-3); ap.add_argument("--lam", type=float, default=0.01)
ap.add_argument("--anchor", type=int, default=0, help="1: predict a residual over the last observed offline penicillin")
ap.add_argument("--val_frac", type=float, default=0.0, help=">0: inner validation split for early stopping and std calibration")
A = ap.parse_args()
torch.set_num_threads(4)
out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
D = np.load(A.hourly, allow_pickle=True).item(); B = D["batches"]; ids = sorted(B)
PI = [i for i, c in enumerate(D["offline"]) if c.startswith("Offline Penicillin")][0]
T_TARGET, K_MIN, K_MAX, NPC = 156.0, 12, 150, 16
EVAL_K = [24, 48, 72, 96, 120, 144]

def target(b):
    t = B[b]["off_t"]; j = int(np.argmin(np.abs(t - T_TARGET))); assert abs(t[j] - T_TARGET) < 1; return B[b]["off_v"][j, PI]
Y = np.array([target(b) for b in ids], np.float32); GRP = np.array([B[b]["group"] for b in ids])
H = int(K_MAX)  # hours of dense data ever needed
ym, ysd = float(Y.mean()), float(Y.std())

# ---------------- features (fit scalers / PCA on training batches only) ----------------
def build(train_idx):
    P = np.concatenate([B[ids[i]]["proc"][:H] for i in train_idx]); pm, ps = np.nanmean(P, 0), np.nanstd(P, 0) + 1e-6
    R = np.concatenate([B[ids[i]]["raman"][:H] for i in train_idx]); R = R[np.isfinite(R).all(1)]
    rsc = StandardScaler().fit(R); pca = PCA(NPC, random_state=0).fit(rsc.transform(R))
    Z = pca.transform(rsc.transform(R)); zs = Z.std(0) + 1e-6
    O = np.concatenate([B[ids[i]]["off_v"] for i in train_idx]); om, osd = O.mean(0), O.std(0) + 1e-6
    feats = {}
    for b in ids:
        p = (np.nan_to_num(B[b]["proc"][:H], nan=0) - pm) / ps
        p = np.where(np.isfinite(B[b]["proc"][:H]), p, 0)
        r = B[b]["raman"][:H]; ok = np.isfinite(r).all(1)
        z = np.zeros((len(r), NPC), np.float32); z[ok] = pca.transform(rsc.transform(r[ok])) / zs
        ot, ov = B[b]["off_t"], (B[b]["off_v"] - om) / osd
        keep = ot < T_TARGET - 1  # never feed the target assay
        feats[b] = dict(dense=np.c_[p, z].astype(np.float32), off_t=ot[keep].astype(np.float32), off_v=ov[keep].astype(np.float32),
                        off_p=B[b]["off_v"][keep, PI].astype(np.float32))
    return feats

def grid_offline(f, T):
    """forward-fill offline values onto the hourly grid (value known from the hour after the sample)."""
    g = np.zeros((T, f["off_v"].shape[1] + 1), np.float32)
    for t, v in zip(f["off_t"], f["off_v"]):
        h = int(math.ceil(t));
        if h < T: g[h:, :-1] = v; g[h:, -1] = 1
    return g

def make_batch(feats, bids, ks, variant):
    n = len(bids); T = max(ks); dd = feats[bids[0]]["dense"].shape[1] + (6 if variant == "grid" else 0)
    dense = np.zeros((n, T, dd), np.float32); dt = np.zeros((n, T), np.float32); dmask = np.ones((n, T), bool)
    maxo = 1 + max(int((feats[b]["off_t"] <= k).sum()) for b, k in zip(bids, ks))
    off = np.zeros((n, maxo, 6), np.float32); ot = np.zeros((n, maxo), np.float32); omask = np.ones((n, maxo), bool)
    for i, (b, k) in enumerate(zip(bids, ks)):
        f = feats[b]; x = f["dense"][:k]
        if variant == "grid": x = np.c_[x, grid_offline(f, len(f["dense"]))[:k]]
        dense[i, :k] = x; dt[i, :k] = (np.arange(k) + 1.0) / T_TARGET; dmask[i, :k] = False
        omask[i, 0] = False; off[i, 0, 5] = 1  # dummy "no assay yet" token at t=0
        if variant == "time":
            m = f["off_t"] <= k; v = f["off_v"][m]; t = f["off_t"][m]
            off[i, 1:1 + len(v), :5] = v; ot[i, 1:1 + len(v)] = t / T_TARGET; omask[i, 1:1 + len(v)] = False
    anc = np.array([(feats[b]["off_p"][feats[b]["off_t"] <= k][-1] - ym) / ysd for b, k in zip(bids, ks)], np.float32)
    tens = lambda a: torch.from_numpy(a)
    return tens(dense), tens(dt), tens(dmask), tens(off), tens(ot), tens(omask), torch.tensor(np.array(ks, np.float32) / T_TARGET), tens(anc)

# ---------------- model ----------------
class TimeEmb(nn.Module):
    def __init__(s, d): super().__init__(); s.lin = nn.Linear(1, 1); s.per = nn.Linear(1, d - 1)
    def forward(s, t): t = t.unsqueeze(-1); return torch.cat([s.lin(t), torch.sin(s.per(t))], -1)

class Net(nn.Module):
    def __init__(s, d_dense, d=32, h=2, dr=0.1, anchor=0):
        super().__init__(); s.anchor = anchor; s.a = nn.Parameter(torch.ones(1))
        s.din, s.oin, s.te = nn.Linear(d_dense, d), nn.Linear(6, d), TimeEmb(d)
        s.mod = nn.Parameter(torch.randn(2, d) * 0.1)
        s.dsa = nn.TransformerEncoderLayer(d, h, 2 * d, dr, batch_first=True)
        s.d2o = nn.MultiheadAttention(d, h, dropout=dr, batch_first=True); s.o2d = nn.MultiheadAttention(d, h, dropout=dr, batch_first=True)
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.q = nn.Parameter(torch.randn(1, 1, d) * 0.1); s.pool = nn.MultiheadAttention(d, h, batch_first=True)
        s.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
    def forward(s, dense, dt, dmask, off, ot, omask, k, anc):
        x = s.din(dense) + s.te(dt) + s.mod[0]
        o = s.oin(off) + s.te(ot) + s.mod[1]
        x = s.dsa(x, src_key_padding_mask=dmask)
        x2 = s.n1(x + s.d2o(x, o, o, key_padding_mask=omask)[0])   # dense queries read the sparse assays
        o2 = s.n2(o + s.o2d(o, x, x, key_padding_mask=dmask)[0])   # assays read the dense stream around them
        kv = torch.cat([x2, o2], 1); km = torch.cat([dmask, omask], 1)
        q = s.q.expand(len(x), -1, -1) + s.te(k.unsqueeze(1))
        z = s.pool(q, kv, kv, key_padding_mask=km)[0][:, 0]
        g, v, a, bb = s.head(z).unbind(-1)
        sp = nn.functional.softplus
        if s.anchor: g = g + s.a * anc
        return g, sp(v) + 1e-3, sp(a) + 1.0 + 1e-3, sp(bb) + 1e-3

def nig_loss(y, g, v, a, b, lam=0.01):
    om = 2 * b * (1 + v)
    nll = 0.5 * torch.log(math.pi / v) - a * torch.log(om) + (a + 0.5) * torch.log(v * (y - g) ** 2 + om) + torch.lgamma(a) - torch.lgamma(a + 0.5)
    return (nll + lam * (y - g).abs() * (2 * v + a)).mean()

def pred_std(v, a, b): return torch.sqrt(b * (1 + v) / (v * (a - 1)))

# ---------------- ridge baseline on identical folds ----------------
def ridge_feats(b, k):
    x = B[b]["proc"][:k]; x = np.nan_to_num(x, nan=np.nanmean(x, 0)); t = np.arange(len(x)); m = B[b]["off_t"] <= k
    return np.concatenate([x.mean(0), x[-6:].mean(0), np.polyfit(t, x, 1)[0], B[b]["off_v"][m][-1] if m.any() else np.zeros(5)])

# ---------------- run ----------------
variants = A.variants.split(","); rows = []; traces = []
t0 = time.time(); fold_no = 0
for rep in range(A.reps):
    for tr, te in StratifiedKFold(A.folds, shuffle=True, random_state=rep).split(Y, GRP):
        fold_no += 1
        if fold_no > A.max_folds: break
        feats = build(tr); ytr = (Y - ym) / ysd
        for k in EVAL_K:  # ridge
            R = np.array([np.nanmean(B[b]["raman"][k - 6:k], 0) for b in ids]); X = np.array([ridge_feats(b, k) for b in ids])
            pca = make_pipeline(StandardScaler(), PCA(10, random_state=0)).fit(R[tr])
            a_, c_ = np.c_[X[tr], pca.transform(R[tr])], np.c_[X[te], pca.transform(R[te])]
            pr = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(a_, Y[tr]).predict(c_)
            rows += [dict(rep=rep, fold=fold_no, variant="ridge", k=k, batch=ids[i], y=float(Y[i]), pred=float(p), std=np.nan) for i, p in zip(te, pr)]
        for var in variants:
            torch.manual_seed(1000 * rep + fold_no); rng = np.random.default_rng(1000 * rep + fold_no)
            d_dense = feats[ids[0]]["dense"].shape[1] + (6 if var == "grid" else 0)
            net = Net(d_dense, A.d, 2, A.dropout, A.anchor); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=A.wd)
            trn, val = tr, np.array([], int)
            if A.val_frac > 0:
                vsel = rng.permutation(len(tr))[:max(4, int(round(A.val_frac * len(tr))))]; val = tr[vsel]; trn = np.delete(tr, vsel)
            best, best_state, scale = np.inf, None, 1.0
            def val_eval():
                net.eval(); vb = [ids[i] for i in val]; zs, nl = [], 0.0
                with torch.no_grad():
                    for k in EVAL_K:
                        o = net(*make_batch(feats, vb, [k] * len(vb), var)); yv = torch.tensor(ytr[val])
                        nl += float(nig_loss(yv, *o, lam=0.0)); zs.append(((yv - o[0]).abs() / pred_std(*o[1:])).numpy())
                net.train(); return nl / len(EVAL_K), np.concatenate(zs)
            steps = A.epochs * math.ceil(len(trn) / 16); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
            net.train(); s = 0
            for ep in range(A.epochs):
                perm = rng.permutation(trn)
                for j in range(0, len(perm), 16):
                    bi = perm[j:j + 16]; ks = list(rng.integers(K_MIN, K_MAX + 1, len(bi)))
                    batch = make_batch(feats, [ids[i] for i in bi], ks, var)
                    loss = nig_loss(torch.tensor(ytr[bi]), *net(*batch))
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); s += 1
                if len(val) and (ep + 1) % 10 == 0:
                    nl, _ = val_eval()
                    if nl < best: best, best_state = nl, {kk: vv.clone() for kk, vv in net.state_dict().items()}
            if best_state is not None:
                net.load_state_dict(best_state); _, z = val_eval(); scale = float(np.quantile(z, 0.8) / 1.2816)
            net.eval()
            with torch.no_grad():
                tb = [ids[i] for i in te]
                for k in EVAL_K:
                    g, v, a, b = net(*make_batch(feats, tb, [k] * len(tb), var))
                    sd = pred_std(v, a, b) * scale
                    rows += [dict(rep=rep, fold=fold_no, variant=var, k=k, batch=bb, y=float(Y[i]), pred=float(gg * ysd + ym), std=float(ss * ysd))
                             for i, bb, gg, ss in zip(te, tb, g.numpy(), sd.numpy())]
                if var == "time":  # hourly trace for the uncertainty-at-assay analysis
                    for k in range(24, 145):
                        g, v, a, b = net(*make_batch(feats, tb, [k] * len(tb), var)); sd = pred_std(v, a, b) * scale
                        traces += [dict(rep=rep, batch=bb, k=k, pred=float(gg * ysd + ym), std=float(ss * ysd), y=float(Y[i]))
                                   for i, bb, gg, ss in zip(te, tb, g.numpy(), sd.numpy())]
            print(f"rep {rep} fold {fold_no} {var}: done ({time.time() - t0:.0f}s) best_val_nll={best:.3f} std_scale={scale:.2f}", flush=True)
    if fold_no > A.max_folds: break

import pandas as pd
df = pd.DataFrame(rows); df.to_csv(out / "predictions.csv", index=False); pd.DataFrame(traces).to_csv(out / "traces.csv", index=False)
def summ(g):
    e = g.pred - g.y; r2 = 1 - (e ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum()
    d = dict(mae_pct=100 * e.abs().mean() / g.y.mean(), r2=r2)
    if g["std"].notna().all():
        from scipy import stats
        z = (e.abs() / g["std"]).to_numpy(); d["cover80"] = (z <= 1.2816).mean(); d["cover95"] = (z <= 1.96).mean()
        d["spearman_std_err"] = stats.spearmanr(g["std"], e.abs()).correlation
    return pd.Series(d)
res = df.groupby(["variant", "k"]).apply(summ, include_groups=False).round(3)
print(res.to_string()); res.to_csv(out / "summary.csv")
json.dump(vars(A), open(out / "args.json", "w"))
