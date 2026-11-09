# Project plan — a hybrid physics–ML FLEXPART

## Idea

FLEXPART (a Lagrangian particle dispersion model) is a chain of physical equations: meteorological
interpolation, particle release, turbulence statistics, the Langevin equation, advection, convection,
removal (wet/dry deposition, chemistry) and concentration sampling.

Most of these are short closed-form formulas: a neural network cannot beat them. A few are expensive
or empirical. The plan is a **hybrid model**: keep the simple equations as exact physics and replace
only the complex or empirical modules with small trained models.

| FLEXPART step | Keep physics or ML | Reason |
|---|---|---|
| Meteorological interpolation | Physics | Exact and cheap |
| Particle release | Physics | Trivial |
| Turbulence statistics (Hanna 1982) | **ML (Phase 3)** | Empirical fits; could be learned from LES or tower data |
| Langevin equation | Physics | Takes σ and τ_L from the turbulence module; keeps the well-mixed condition |
| Advection | Physics | One multiply-add |
| **Convection (Emanuel & Živković-Rothman)** | **ML (Phases 1–2)** | Heaviest per-step module; decides how ocean gases (e.g. bromoform) reach the tropopause |
| Wet scavenging | **ML (Phase 3)** | Empirical (Λ = A·I^B); could be learned from observations |
| Dry deposition, decay, OH loss | Physics | Simple exponentials and resistance formulas |
| Concentration sampling | Physics | Exact counting |

## Phases

| Phase | Work | Status |
|---|---|---|
| **1** | Offline ML emulator of FLEXPART's convection scheme, trained on FLEXPART's own Fortran code | **Complete** (v6, see below) |
| 2 | Retrain on FLEXPART's model levels with more varied training regions; couple the emulator into FLEXPART (TorchScript model called from Fortran via FTorch); long online stability runs | Next |
| 3 | ML turbulence-statistics and wet-scavenging modules trained on better data (LES, observations), aiming to be *more accurate* than the original fits | Planned |
| 4 | Validate the hybrid model against ship and aircraft observations (e.g. bromoform transport) | Planned |

## Phase 1 — plan and outcome

### Research question
Can a small set of trained models reproduce the convective redistribution matrix of FLEXPART's
Emanuel–Živković-Rothman scheme accurately enough to be coupled into FLEXPART?

### Objectives and outcome

| # | Objective | Outcome |
|---|---|---|
| 1 | Training set of ≥ 100,000 columns paired with the scheme's redistribution matrix | **Done**: 137,760 ERA5 columns (tropical Indian Ocean, 2019–2023), labelled by FLEXPART's real scheme; v6 keeps the ~98 % with surface pressure above 1000 hPa |
| 2 | Small models (each ≲ 1 MB) | **Done**: four models (gate, cloud top, redistribution, cloud-base mass flux), ≈ 1.9 MB in total |
| 3 | Accuracy targets on unseen years and regions | **Met** on the 2023 test year; in the tropical Atlantic met for per-column structure, while the mass-flux profile shows a regime-shift gap (see below) |
| 4 | Speed against the original scheme | Measured: **1.3–1.6×** faster on CPU, far below the 10× hoped for. The scheme is cheaper than expected; the value of the project lies in accuracy and structure, not speed |
| 5 | TorchScript export for Phase 2 | **Done** |

### Gates

- **Gate 1 — true teacher labels.** Passed: `CONVECT` and `TLIFT` were extracted from FLEXPART 11's
  `conv_mod.f90`, the `ERROR STOP` ("Convection reaches top level of input data") was patched into a
  flag, and the scheme was compiled as a standalone library callable from Python.
- **Gate 2 — accuracy targets.** Passed on real ERA5 data (v5, v6). In the held-out region, per-column
  structure generalises (ratio ≤ 1.12×); the mass-flux profile does not fully (ratio 2.1× at levels carrying transport).

### Validation targets (Phase 1)

| Check | Target | v5, Indian Ocean 2023 | v5, Atlantic 2023 | v6, Indian Ocean 2023 | v6, Atlantic 2023 |
|---|---|---|---|---|---|
| Gate F1 | ≥ 0.95 | 0.989 | 0.976 | 0.994 | 0.981 |
| Matrix mean absolute error | ≤ 0.01 | 1.8 × 10⁻⁵ | — | 1.7 × 10⁻⁵ | — |
| Mass flux, worst significant level | within 10 % | 10.9 % | 7.9 % | 4.7 % | 11.2 % (level with 1 % of peak flux) |
| Mass flux, worst level with ≥ 10 % of peak flux | within 10 % | — | — | 4.6 % | 9.5 % |
| Off-diagonal error, median | — | 34 % | 38 % | 29 % | 33 % |
| Particle test (Wasserstein, median) | small | 0.014 hPa | 0.013 hPa | 0.013 hPa | 0.011 hPa |
| Mass conservation (rows sum to 1) | exact | exact | exact | exact | exact |
| Held-out / training-region error | ≤ 1.5× | — | ≤ 1.13× | — | structure ≤ 1.12×; mass flux 2.1× |
| Speed | ≥ 10× | 1.4× | — | timing varies (1.3–2.4×) | — |

v5 and v6 test sets differ slightly: v6 excludes columns with surface pressure below 1000 hPa (mostly land).

### Remaining Phase 1 items
- Gate threshold tested (0.5 → 0.2, no retraining): 91–98 % of missed columns are weak convection; 0.2 recovers
  ~40 % of misses, and transport checks change by < 0.3 points. Recommended threshold 0.3. Remaining misses need a
  better-trained gate (more weight on weak convection).
- Real surface pressure: **done in v6** (input feature and bottom layer boundary; below-ground columns excluded).
- Atlantic mass-flux gap: diagnosed as a **convection-regime shift** (more shallow convection under higher
  surface pressure than in the Indian Ocean training data), not a surface-pressure input problem
  (clipping the input changes the error only 12.3 % → 11.9 %). Carried into Phase 2: train on several regions.
- Model-level input (terrain-following, so land columns can be kept): carried into Phase 2.

### Risks that materialised, and what was done

| Risk | What happened | Response |
|---|---|---|
| Real scheme hard to compile alone | Compiled; `ERROR STOP` killed the Python session on short columns | Patched into a flag (`iflag = −1`); columns extended to 10 hPa |
| Copernicus download limits | Only one queued request per dataset; ~1.5 h per year | Sequential requests |
| Averaged metrics hide errors | Matrix MAE looked excellent while single columns were blurred | Added per-column off-diagonal error and strongest-column plots |
| Flux loss destabilises training | From the start, the network collapsed to "no air moves" | Flux term used only as a fine-tuning stage |
| Cloud-top mistakes | Hard masking wiped out a column when the top was underestimated | Cautious cloud top (95 % confidence) + training slack |
| Region shift | Atlantic convection shallower than in the training region; mass flux overestimated 6–9 % above the boundary layer | Diagnosed (v6); remedy: more varied training regions in Phase 2 |
