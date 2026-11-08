# === Held-out region test: v5 emulator (trained on the Indian Ocean) on the tropical Atlantic ===
import os, json, numpy as np, torch
import matplotlib.pyplot as plt

needed = ["P_LEVELS", "NLEV", "LAYER_DP", "LAYER_W", "CFG", "flexpart_column",
          "parcel_diagnostics", "load_era5_columns"]
missing = [n for n in needed if n not in globals()]
if missing:
    raise SystemExit(f"Missing {missing}: run the notebook's cells from the top down to and including "
                     "Section 3 (Atmospheric columns), then run this cell again.")

DRIVE = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else os.path.abspath(".")
V5_DIR = os.path.join(DRIVE, "flexpart_phase1_v5") if os.path.isdir(os.path.join(DRIVE, "flexpart_phase1_v5")) else OUT_DIR
HELDOUT = os.path.join(DRIVE, "era5_heldout", "era5_atlantic_2023.nc")

# 1. Atlantic columns + real FLEXPART labels (same random spin-up of cbmf as in training)
T_a, q_a, ps_a, _ = load_era5_columns(HELDOUT)
rng_a = np.random.default_rng(123)
N = len(T_a)
M_a = np.empty((N, NLEV, NLEV)); cbo_a = np.zeros(N); cbn_a = np.zeros(N); fl_a = np.zeros(N, int)
for c in range(N):
    cb = 0.0
    for _ in range(rng_a.integers(0, CFG["max_spinup_calls"] + 1)):
        _, cb, fl, _ = flexpart_column(T_a[c], q_a[c], cb)
        if fl == -1:
            break
    M_a[c], cbn_a[c], fl_a[c], _ = flexpart_column(T_a[c], q_a[c], cb)
    cbo_a[c] = cb
ok = fl_a != -1
T_a, q_a, ps_a, M_a, cbo_a, cbn_a, fl_a = T_a[ok], q_a[ok], ps_a[ok], M_a[ok], cbo_a[ok], cbn_a[ok], fl_a[ok]
conv_a = np.isin(fl_a, [1, 4]) & ~np.all(np.isclose(M_a, np.eye(NLEV)[None]), axis=(1, 2))
act = np.abs(M_a - np.eye(NLEV)[None]) > 1e-7
lev_act = act.any(axis=2) | act.any(axis=1)
top_a = np.where(lev_act.any(axis=1), NLEV - 1 - lev_act[:, ::-1].argmax(axis=1), 0)
print(f"Atlantic: {len(T_a):,} columns kept ({np.sum(~ok)} too short) | convective {conv_a.mean():.1%}")

# 2. Features exactly as in training, then the saved v5 emulator (TorchScript, normalisation built in)
d = parcel_diagnostics(T_a, q_a)
X_a = np.concatenate([T_a, q_a * 1000.0, ps_a[:, None], d["cape"][:, None] / 1000.0, d["cin"][:, None] / 100.0,
                      d["lcl_p"][:, None] / 1000.0, cbo_a[:, None] * 100.0], axis=1).astype(np.float32)
emu = torch.jit.load(os.path.join(V5_DIR, "convection_emulator_torchscript.pt"))
with torch.no_grad():
    outs = [emu(torch.tensor(X_a[i:i + 4096])) for i in range(0, len(X_a), 4096)]
prob = np.concatenate([o[0].numpy() for o in outs]); topc = np.concatenate([o[1].numpy() for o in outs])
Mh = np.concatenate([o[2].numpy() for o in outs]).astype(np.float64); cbh = np.concatenate([o[3].numpy() for o in outs])

# 3. The same checks as in the notebook
def upward_flux(M):
    m = LAYER_DP[None, :, None]
    return np.stack([(m[:, :k + 1] * M[:, :k + 1, k + 1:]).sum(axis=(1, 2)) for k in range(NLEV - 1)], axis=1)
pred = prob > 0.5
tp, fp, fn = (pred & conv_a).sum(), (pred & ~conv_a).sum(), (~pred & conv_a).sum()
c = conv_a
ft, fm = upward_flux(M_a[c]).mean(0), upward_flux(Mh[c]).mean(0)
sig = ft > 0.01 * ft.max()
off = ~np.eye(NLEV, dtype=bool)[None]
num = (np.abs(Mh[c] - M_a[c]) * off).sum(axis=(1, 2)); den = (np.abs(M_a[c]) * off).sum(axis=(1, 2))
okd = den > 1e-9; rel = num[okd] / den[okd]
strong = cbn_a[c][okd] >= np.percentile(cbn_a[c][okd], 90)
cdf = lambda M: np.cumsum((LAYER_W[None, :, None] * M).sum(axis=1), axis=1)
w1 = (np.abs(cdf(Mh[c]) - cdf(M_a[c])) * LAYER_DP[None, :]).sum(axis=1)
cbt, cbm = cbn_a[c], cbh[c]
_v = np.sum((cbt - cbt.mean()) ** 2)
r2 = 1 - np.sum((cbm - cbt) ** 2) / _v if _v > 1e-12 else float("nan")

res = {
    "region": "tropical Atlantic 2023 (never seen in training)",
    "convective_share": float(conv_a.mean()),
    "gate_F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
    "cautious_top_at_or_above_true": float(np.mean(topc[c] >= top_a[c])),
    "mass_flux_max_relative_error": float((np.abs(fm - ft)[sig] / ft[sig]).max()),
    "offdiag_relative_error_median": float(np.median(rel)),
    "offdiag_relative_error_median_strongest_10pct": float(np.median(rel[strong])),
    "particle_W1_hPa_median": float(np.median(w1)),
    "cbmf_R2": float(r2),
    "row_sums_equal_1": bool(np.allclose(Mh.sum(axis=2), 1.0, atol=1e-4)),
}
with open(os.path.join(V5_DIR, "heldout_atlantic_results.json"), "w") as f:
    json.dump(res, f, indent=2)

# 4. Side by side with the Indian Ocean test year (from v5's results)
v5 = json.load(open(os.path.join(V5_DIR, "validation_results.json")))
print(f"\n{'check':48s} {'Indian Ocean 2023':>18s} {'Atlantic 2023':>14s}")
for k in ["gate_F1", "cautious_top_at_or_above_true", "mass_flux_max_relative_error",
          "offdiag_relative_error_median", "offdiag_relative_error_median_strongest_10pct",
          "particle_W1_hPa_median", "cbmf_R2"]:
    print(f"{k:48s} {v5.get(k, float('nan')):18.4f} {res[k]:14.4f}")
print(f"{'convective share':48s} {'':>18s} {res['convective_share']:14.1%}")
print("Rows sum to 1:", res["row_sums_equal_1"])

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(ft, P_LEVELS[:-1], "k-", label="FLEXPART"); ax[0].plot(fm, P_LEVELS[:-1], "C1--", label="emulator")
ax[0].invert_yaxis(); ax[0].set_yscale("log")
ax[0].set(title="Atlantic: mean upward mass flux", xlabel="hPa of air crossing level", ylabel="pressure (hPa)"); ax[0].legend()
ax[1].hist(rel, bins=50, range=(0, 2)); ax[1].set(title="Atlantic: off-diagonal relative error", xlabel="relative error", ylabel="columns")
ax[2].scatter(cbt, cbm, s=4, alpha=0.4); lim = max(cbt.max(), cbm.max(), 1e-6)
ax[2].plot([0, lim], [0, lim], "k--", lw=1); ax[2].set(title=f"Atlantic: cbmf_new (R² = {r2:.3f})", xlabel="FLEXPART", ylabel="emulator")
plt.tight_layout(); plt.savefig(os.path.join(V5_DIR, "heldout_atlantic_summary.png"), dpi=130); plt.show()
