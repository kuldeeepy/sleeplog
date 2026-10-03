"""Train the motion sleep/wake model on PhysioNet sleep-accel (Walch 2019) and export it as JSON.

Counts are normalised per night (median / spread of log counts), so the model does not care
about the phone's sensor scale or how much the mattress damps movement.
Run: <val_motion venv>/python train_model.py <path to epochs_dead0.02.pkl>
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

WINDOWS = (3, 11, 21, 41)


def features(counts):
    x = np.log1p(np.asarray(counts, float))
    lo, mid, hi = np.percentile(x, [10, 50, 90])
    s = pd.Series((x - mid) / max(hi - lo, 1e-6))
    f = [s]
    for k in WINDOWS:
        f += [s.rolling(k, center=True, min_periods=1).mean(), s.rolling(k, center=True, min_periods=1).max()]
    f.append(s.rolling(11, center=True, min_periods=1).std().fillna(0))
    f.append(pd.Series(np.linspace(0, 1, len(s))))
    return np.column_stack(f)


df = pd.read_pickle(sys.argv[1])
df = df[(df.stage >= 0) & df.valid]
nights = []
for cond in ("mat0.3", "mat0.1"):  # simulated floor-mattress signals
    for sid, g in df.groupby("sid"):
        nights.append((sid, features(g[f"cnt_{cond}"].values), (g.stage > 0).values))

# leave-one-subject-out check, threshold tuned for kappa
sids = sorted({n[0] for n in nights})
kap, spec, tst = [], [], []
for s in sids:
    tr = [n for n in nights if n[0] != s]
    m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(np.vstack([n[1] for n in tr]), np.concatenate([n[2] for n in tr]))
    ths = np.linspace(0.2, 0.8, 25)
    th = max(ths, key=lambda t: np.mean([cohen_kappa_score(n[2], m.predict_proba(n[1])[:, 1] >= t) for n in tr[::4]]))
    for n in (n for n in nights if n[0] == s):
        p = m.predict_proba(n[1])[:, 1] >= th
        kap.append(cohen_kappa_score(n[2], p)); spec.append((~p & ~n[2]).sum() / max((~n[2]).sum(), 1))
        tst.append((p.sum() - n[2].sum()) / 2)
print(f"LOSO kappa {np.mean(kap):.2f}  wake-catch {np.mean(spec):.2f}  total-sleep err {np.mean(tst):+.0f}±{np.std(tst):.0f} min")

m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(np.vstack([n[1] for n in nights]), np.concatenate([n[2] for n in nights]))
th = max(np.linspace(0.2, 0.8, 25), key=lambda t: np.mean([cohen_kappa_score(n[2], m.predict_proba(n[1])[:, 1] >= t) for n in nights]))
json.dump({"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "threshold": float(th),
           "windows": WINDOWS, "trained_on": "PhysioNet sleep-accel, 31 subjects, simulated floor mattress"},
          open("motion_model.json", "w"), indent=1)
print("saved motion_model.json, threshold", round(th, 3))
