# === Why did the Atlantic mass-flux error rise in v6? Surface pressure check + per-level flux error ===
import os, glob, json, numpy as np, torch
import matplotlib.pyplot as plt

needed = ["P_LEVELS", "PH_LEVELS", "NLEV", "LAYER_DP", "CFG", "flexpart_column", "parcel_diagnostics",
          "load_era5_columns", "drop_below_ground", "ERA5_DIR", "HELDOUT_FILE"]
missing = [n for n in needed if n not in globals()]
if missing:
    raise SystemExit(f"Missing {missing}: run the v6 notebook's cells from the top down to and including "
                     "Section 3 (Atmospheric columns), then run this cell again.")

DRIVE = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else os.path.abspath(".")
V6_DIR = os.path.join(DRIVE, "flexpart_phase1_v6") if os.path.isdir(os.path.join(DRIVE, "flexpart_phase1_v6")) else OUT_DIR
PS_COL = 2 * NLEV                     # surface pressure position in the 69 input features
files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_*.nc")))
train_files, test_file = files[:-2], files[-1]

def load(path, label):
    T, q, ps, _ = load_era5_columns(path)
    T, q, ps, _ = drop_below_ground(T, q, ps, None, label=f"{label}: ")
    return T, q, ps

# ---------- 1. Surface-pressure distributions ----------
print("Loading surface pressure ...")
ps_train = np.concatenate([load(f, os.path.basename(f))[2] for f in train_files])
Ti, qi, psi = load(test_file, "Indian Ocean test")
Ta, qa, psa = load(HELDOUT_FILE, "Atlantic")
lo, hi = np.percentile(ps_train, [1, 99])
print(f"\n{'surface pressure (hPa)':26s} {'1%':>7} {'5%':>7} {'50%':>7} {'95%':>7} {'99%':>7}")
for name, p in [("Indian Ocean train 2019-21", ps_train), ("Indian Ocean test 2023", psi), ("Atlantic 2023", psa)]:
    print(f"{name:26s} " + " ".join(f"{v:7.1f}" for v in np.percentile(p, [1, 5, 50, 95, 99])))
out_a = (psa < lo) | (psa > hi)
print(f"\nAtlantic columns outside the training 1–99 % range ({lo:.1f}–{hi:.1f} hPa): {out_a.mean():.1%} "
      f"(Indian Ocean test: {np.mean((psi < lo) | (psi > hi)):.1%})")

# ---------- 2. FLEXPART labels + emulator predictions ----------
def label(T, q, ps, seed):
    r = np.random.default_rng(seed)
    N = len(T); M = np.empty((N, NLEV, NLEV)); cbo = np.zeros(N); fl = np.zeros(N, int)
    for c in range(N):
        cb = 0.0
        for _ in range(r.integers(0, CFG["max_spinup_calls"] + 1)):
            _, cb, f, _ = flexpart_column(T[c], q[c], cb, ps=ps[c])
            if f == -1:
                break
        M[c], _, fl[c], _ = flexpart_column(T[c], q[c], cb, ps=ps[c])
        cbo[c] = cb
    ok = fl != -1
    conv = ok & np.isin(fl, [1, 4]) & ~np.all(np.isclose(M, np.eye(NLEV)[None]), axis=(1, 2))
    return M, cbo, conv, ok

def features(T, q, ps, cbo):
    d = parcel_diagnostics(T, q)
    return np.concatenate([T, q * 1000.0, ps[:, None], d["cape"][:, None] / 1000.0, d["cin"][:, None] / 100.0,
                           d["lcl_p"][:, None] / 1000.0, cbo[:, None] * 100.0], axis=1).astype(np.float32)

emu = torch.jit.load(os.path.join(V6_DIR, "convection_emulator_torchscript.pt")).eval()
@torch.no_grad()
def emulate(X):
    return np.concatenate([emu(torch.tensor(X[i:i + 4096]))[2].numpy() for i in range(0, len(X), 4096)]).astype(np.float64)

def upward_flux(M):
    m = LAYER_DP[None, :, None]
    return np.stack([(m[:, :k + 1] * M[:, :k + 1, k + 1:]).sum(axis=(1, 2)) for k in range(NLEV - 1)], axis=1)

print("\nLabelling with FLEXPART (about a minute per region) ...")
sets = {}
for name, (T, q, ps), seed in [("Indian Ocean 2023", (Ti, qi, psi), 11), ("Atlantic 2023", (Ta, qa, psa), 12)]:
    M, cbo, conv, ok = label(T, q, ps, seed)
    X = features(T, q, ps, cbo)
    sets[name] = dict(M=M, X=X, conv=conv, ps=ps, Mh=emulate(X))

# ---------- 3. Per-level mass-flux error ----------
def level_table(M, Mh):
    ft, fm = upward_flux(M).mean(0), upward_flux(Mh).mean(0)
    sig = ft > 0.01 * ft.max()
    rel = np.where(sig, np.abs(fm - ft) / np.where(ft > 0, ft, 1), np.nan)
    return ft, fm, rel, sig

print("\nMean upward mass flux across each level interface (convective columns; * = worst significant level)")
print(f"{'interface':>10} | {'Indian: FLEXPART':>16} {'emulator':>9} {'error':>7} | {'Atlantic: FLEXPART':>18} {'emulator':>9} {'error':>7}")
tabs = {n: level_table(s["M"][s["conv"]], s["Mh"][s["conv"]]) for n, s in sets.items()}
worst = {n: int(np.nanargmax(t[2])) for n, t in tabs.items()}
for k in range(NLEV - 1):
    (fti, fmi, ri, si), (fta, fma, ra, sa) = tabs["Indian Ocean 2023"], tabs["Atlantic 2023"]
    if not (si[k] or sa[k]):
        continue
    fmt = lambda r, s, w: (f"{r:6.1%}" if s else "   -  ") + ("*" if w else " ")
    print(f"{PH_LEVELS[k + 1]:7.1f} hPa | {fti[k]:16.4f} {fmi[k]:9.4f} {fmt(ri[k], si[k], k == worst['Indian Ocean 2023'])} | "
          f"{fta[k]:18.4f} {fma[k]:9.4f} {fmt(ra[k], sa[k], k == worst['Atlantic 2023'])}")
for n in sets:
    k = worst[n]
    print(f"{n}: worst level at {PH_LEVELS[k + 1]:.1f} hPa, error {tabs[n][2][k]:.1%} "
          f"(flux there = {tabs[n][0][k] / tabs[n][0].max():.0%} of the column maximum)")

# ---------- 4. Direct test: Atlantic inside vs outside the training surface-pressure range ----------
A = sets["Atlantic 2023"]; c = A["conv"]
def worst_err(M, Mh):
    _, _, rel, _ = level_table(M, Mh); return np.nanmax(rel)
inside, outside = c & ~out_a, c & out_a
print(f"\nAtlantic mass-flux error, columns INSIDE the training range : {worst_err(A['M'][inside], A['Mh'][inside]):.1%} "
      f"({inside.sum():,} columns)")
if outside.sum() >= 200:
    print(f"Atlantic mass-flux error, columns OUTSIDE the training range: {worst_err(A['M'][outside], A['Mh'][outside]):.1%} "
          f"({outside.sum():,} columns)")
else:
    print(f"Only {outside.sum()} convective Atlantic columns outside the training range — too few for a separate error.")
Xc = A["X"].copy(); Xc[:, PS_COL] = np.clip(Xc[:, PS_COL], lo, hi)
Mh_clip = emulate(Xc)
print(f"Atlantic mass-flux error with the surface-pressure INPUT clipped to the training range: "
      f"{worst_err(A['M'][c], Mh_clip[c]):.1%}   (unclipped: {worst_err(A['M'][c], A['Mh'][c]):.1%})")

# ---------- 5. Plots ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
bins = np.linspace(min(ps_train.min(), psa.min()), max(ps_train.max(), psa.max()), 60)
ax[0].hist(ps_train, bins=bins, density=True, alpha=0.5, label="Indian Ocean train 2019–21")
ax[0].hist(psi, bins=bins, density=True, histtype="step", lw=1.5, label="Indian Ocean test 2023")
ax[0].hist(psa, bins=bins, density=True, histtype="step", lw=1.5, label="Atlantic 2023")
ax[0].axvline(lo, color="grey", ls=":"); ax[0].axvline(hi, color="grey", ls=":")
ax[0].set(title="Surface pressure (dotted: training 1–99 %)", xlabel="hPa", ylabel="density"); ax[0].legend(fontsize=8)
for n, st in [("Indian Ocean 2023", "-"), ("Atlantic 2023", "--")]:
    ax[1].plot(tabs[n][2] * 100, PH_LEVELS[1:NLEV], st, marker=".", label=n)
ax[1].axvline(10, color="grey", ls=":"); ax[1].invert_yaxis(); ax[1].set_yscale("log")
ax[1].set(title="Mass-flux error per level (significant levels)", xlabel="relative error (%)", ylabel="pressure (hPa)")
ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(V6_DIR, "flux_error_diagnostic.png"), dpi=130); plt.show()
