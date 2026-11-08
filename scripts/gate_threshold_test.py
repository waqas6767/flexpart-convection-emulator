# === Gate threshold test on the saved v5 model (no retraining) ===
import os, glob, numpy as np, torch

needed = ["P_LEVELS", "NLEV", "LAYER_DP", "CFG", "flexpart_column", "parcel_diagnostics", "load_era5_columns"]
missing = [n for n in needed if n not in globals()]
if missing:
    raise SystemExit(f"Missing {missing}: run the v5 notebook's cells from the top down to and including "
                     "Section 3 (Atmospheric columns), then run this cell again.")

DRIVE = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else os.path.abspath(".")
V5_DIR = os.path.join(DRIVE, "flexpart_phase1_v5") if os.path.isdir(os.path.join(DRIVE, "flexpart_phase1_v5")) else OUT_DIR
TEST_SETS = {
    "Indian Ocean 2023": sorted(glob.glob(os.path.join(DRIVE, "era5_phase1", "era5_*.nc")))[-1],   # the test year
    "Atlantic 2023": os.path.join(DRIVE, "era5_heldout", "era5_atlantic_2023.nc"),
}
THRESHOLDS = [0.5, 0.4, 0.3, 0.2]
EYE = np.eye(NLEV)

def label_and_features(path, seed):
    """Real FLEXPART labels (with the same random cbmf spin-up as in training) + the 69 input features."""
    T, q, ps, _ = load_era5_columns(path)
    r = np.random.default_rng(seed)
    N = len(T); M = np.empty((N, NLEV, NLEV)); cbo = np.zeros(N); cbn = np.zeros(N); fl = np.zeros(N, int)
    for c in range(N):
        cb = 0.0
        for _ in range(r.integers(0, CFG["max_spinup_calls"] + 1)):
            _, cb, f, _ = flexpart_column(T[c], q[c], cb)
            if f == -1:
                break
        M[c], cbn[c], fl[c], _ = flexpart_column(T[c], q[c], cb)
        cbo[c] = cb
    ok = fl != -1
    T, q, ps, M, cbo, cbn, fl = T[ok], q[ok], ps[ok], M[ok], cbo[ok], cbn[ok], fl[ok]
    conv = np.isin(fl, [1, 4]) & ~np.all(np.isclose(M, EYE[None]), axis=(1, 2))
    d = parcel_diagnostics(T, q)
    X = np.concatenate([T, q * 1000.0, ps[:, None], d["cape"][:, None] / 1000.0, d["cin"][:, None] / 100.0,
                        d["lcl_p"][:, None] / 1000.0, cbo[:, None] * 100.0], axis=1).astype(np.float32)
    return X, M, conv, cbo, cbn

emu = torch.jit.load(os.path.join(V5_DIR, "convection_emulator_torchscript.pt")).eval()
# The traced redistribution sub-model may store its two inputs in either order: check its signature
_sig = emu.redist.code.split(")")[0]
TOP_FIRST = _sig.find("top") < _sig.find("input") if "input" in _sig else False

@torch.no_grad()
def raw_predictions(X):
    """Gate probability and the UN-gated matrix and cbmf from the saved v5 sub-models."""
    P, Ms, CB = [], [], []
    for i in range(0, len(X), 4096):
        xr = torch.tensor(X[i:i + 4096])
        x = (xr - emu.mean) / emu.std
        cdf = torch.cumsum(torch.softmax(emu.topnet(x), dim=-1), dim=-1)
        top = (cdf < CFG["top_confidence"]).sum(dim=-1).clamp(max=NLEV - 1).long()   # cautious cloud top
        P.append(torch.sigmoid(emu.gate(x)).numpy())
        Ms.append((emu.redist(top, x) if TOP_FIRST else emu.redist(x, top)).exp().numpy())
        CB.append((emu.cbnet(x) / 100.0).numpy())
    return np.concatenate(P), np.concatenate(Ms).astype(np.float64), np.concatenate(CB)

def upward_flux(M):
    m = LAYER_DP[None, :, None]
    return np.stack([(m[:, :k + 1] * M[:, :k + 1, k + 1:]).sum(axis=(1, 2)) for k in range(NLEV - 1)], axis=1)

OFF = ~np.eye(NLEV, dtype=bool)[None]
all_rows = []
for name, path in TEST_SETS.items():
    print(f"\n=== {name} ===")
    X, M, conv, cbo, cbn = label_and_features(path, seed=7)
    prob, Mraw, cbraw = raw_predictions(X)
    # check: at 0.5 the raw sub-models must reproduce the exported model exactly
    with torch.no_grad():
        ref = emu(torch.tensor(X[:512]))[2].numpy()
    same = np.allclose(np.where((prob[:512] > 0.5)[:, None, None], Mraw[:512], EYE[None]), ref, atol=1e-5)
    print(f"{len(X):,} columns, {conv.sum():,} convective ({conv.mean():.1%}) | sub-models match exported model: {same}")

    ft = upward_flux(M[conv]).mean(0); sig = ft > 0.01 * ft.max()
    weak_cut = np.percentile(cbn[conv], 25)                    # 'weak convection' = lowest quarter of cbmf_new
    print(f"{'threshold':>9} {'F1':>6} {'missed':>7} {'weak missed':>11} {'false alarms':>12} "
          f"{'flux err':>8} {'off-diag med':>12} {'cols at 100%':>12} {'cbmf R2':>8}")
    for t in THRESHOLDS:
        on = prob > t
        Mf = np.where(on[:, None, None], Mraw, EYE[None])
        cbf = np.where(on, cbraw, cbo)
        tp, fp, fn = (on & conv).sum(), (on & ~conv).sum(), (~on & conv).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        weak_missed = (~on & conv & (cbn <= weak_cut)).sum()
        fm = upward_flux(Mf[conv]).mean(0)
        flux_err = (np.abs(fm - ft)[sig] / ft[sig]).max()
        num = (np.abs(Mf[conv] - M[conv]) * OFF).sum(axis=(1, 2)); den = (M[conv] * OFF).sum(axis=(1, 2))
        okd = den > 1e-9; rel = num[okd] / den[okd]
        at100 = np.mean(np.isclose(rel, 1.0, atol=1e-3))       # columns where no air was moved at all
        cbt, cbm = cbn[conv], cbf[conv]
        v = np.sum((cbt - cbt.mean()) ** 2)
        r2 = 1 - np.sum((cbm - cbt) ** 2) / v if v > 1e-12 else float("nan")
        print(f"{t:9.1f} {f1:6.3f} {fn:7d} {weak_missed:11d} {fp:12d} {flux_err:8.1%} {np.median(rel):12.1%} "
              f"{at100:12.1%} {r2:8.3f}")
        all_rows.append(dict(region=name, threshold=t, F1=float(f1), missed=int(fn), weak_missed=int(weak_missed),
                             false_alarms=int(fp), flux_err=float(flux_err), offdiag_median=float(np.median(rel)),
                             cols_at_100pct=float(at100), cbmf_R2=float(r2)))

import json
with open(os.path.join(V5_DIR, "gate_threshold_test.json"), "w") as f:
    json.dump(all_rows, f, indent=2)
print("\nSaved:", os.path.join(V5_DIR, "gate_threshold_test.json"))
