# CLAUDE.md

This is a heavy-quarkonium spectroscopy calculator. You give it two quark masses
and a Cornell potential, it solves the Schrödinger equation in a Gaussian basis,
and out come meson masses, decay widths, and how far each one sits from experiment.
That's the whole story. Everything below is detail.

## The one rule about Python

There's a conda env called `400`. Use it. Always:

```bash
conda run -n 400 python scripts/run_spectrum.py
```

Bare `python` / `python3` grabs the wrong interpreter and you'll burn ten minutes
confused about a missing `scipy`. Don't. The deps are just `numpy`, `scipy`,
`matplotlib`, `pandas`.

## Commands

The whole pipeline is one command:

```bash
conda run -n 400 python scripts/run_spectrum.py
```

The first run is slow — minutes — because it fits Cornell parameters for each
sector. After that it's fast: fits get cached to `results/<sector>/params.csv` and
reloaded, so the second run is cheap. That caching is deliberate, not an accident.

Want a fresh fit? Delete the cache and rerun — it refits only what's missing:

```bash
rm results/bb/params.csv   # next run_spectrum.py refits just bb
```

Note: if you change the *fit objective* itself (the error model, the weights, the
decay residuals), the cached params are stale — delete them (or `rm -rf results`)
so the refit actually happens.

Benchmark against PDG (needs `run_spectrum.py` to have run first — it reads the
per-sector `errors.csv` files):

```bash
conda run -n 400 python scripts/test_spectra.py
```

Check the fit for overfitting (train/test split on the mass levels, fast — uses a
small radial grid since the GEM eigenvalues don't depend on it):

```bash
conda run -n 400 python scripts/cross_validate.py
```

Rebuild the human-readable writeup at `REPORT.md` from whatever's in `results/`:

```bash
conda run -n 400 python scripts/generate_report.py
```

Make every analysis figure (vector PDF + PNG into `figures/`). Reads only the
CSVs under `results/` plus the curated literature values in
`scripts/literature_data.py`, so it is fast and does *not* re-run the fit — run
`run_spectrum.py` (and `cross_validate.py` for the overfitting panel) first so the
CSVs exist:

```bash
conda run -n 400 python scripts/make_figures.py
```

Old QHO sanity check, kept around for validation:

```bash
conda run -n 400 python qho/plot_qho.py
```

## Where things live

The physics is a package under `src/`; the scripts that drive it are under
`scripts/`. `run_spectrum.py` bolts `src/` onto `sys.path` itself, so you run the
scripts directly — there's no install step.

```text
src/quarkonia/
  gem_solver.py       the eigensolver — this is the heart
  fitter.py           fits Cornell params (masses + leptonic widths) to PDG data
  observables.py      relativistic corrections + |R(0)|^2
  decay_models.py     decay widths
  metrics.py          formatting, CSV export, the consolidated report
  pdg_loader.py       reads experimental masses/widths/errors from the PDG SQLite
  paths.py            single source of truth for everything under results/
scripts/
  run_spectrum.py     main entry point, wires it all together
  test_spectra.py     PDG benchmark pass
  cross_validate.py   train/test split — overfitting diagnostic
  generate_report.py  results/ CSVs -> REPORT.md
  make_figures.py     results/ CSVs + literature_data.py -> figures/*.pdf,*.png
  literature_data.py  curated theory/experiment values from the literature
tex/
  apssamp.tex         the full REVTeX paper (the long-form final report)
prestex/
  presentation.tex    the beamer final-presentation deck (long form, ~15-30 min)
```

The figures, the LaTeX paper under `tex/`, and the beamer deck under `prestex/`
are the submission artifacts. `make_figures.py` is the only figure generator; the
old `eps.gen.py` was retired when the per-sector `results/<sector>/` layout
replaced the flat one.

**The final presentation** lives in `prestex/presentation.tex` — a self-contained
beamer deck (`aspectratio=169`, stock **Madrid** theme to match the earlier mock
deck) that pulls its figures straight from `../figures/*.pdf`, so it has no
dependency on the slow fit. Build it with plain LaTeX (no conda — that rule is
Python-only):

```bash
cd prestex && latexmk -pdf presentation.tex   # or: pdflatex presentation.tex (x2)
```

This is the **long-form** final talk (~15-30 min, ~24 content slides across 7
`\section`s: Motivation, GEM, Spin corrections, Fitting/Uncertainties, Mass
Spectra, Decays, Validation/Outlook). It is the *comprehensive* version that walks
through every method in the pipeline — not the earlier 3-min/5-slide mock. Writing
style mirrors the student's own: noun-phrase frame titles, **bold lead-in** bullets,
honest numbers (bb RMS 7.0 MeV, cc 28.8 MeV, the η_c→γγ ×2.4 overshoot called out as
an NR-limit artefact). If you trim it back to a short course-limit version, branch —
don't overwrite this one.

**Submission naming** (everything ships from `submits/`):

```text
submits/
  400_prop_2555365_leventkaanoguz.pdf   proposal
  400_inter_2555365_leventkaanoguz.pdf  interim report
  400_revtex_2555365.pdf                final paper (copy of tex/apssamp.pdf)
  400_fipres_2555365.pdf                final presentation (copy of prestex/presentation.pdf)
```

After recompiling either LaTeX source, copy the fresh PDF into `submits/` under its
`400_<kind>_2555365*.pdf` name — the graders read `submits/`, not the build dirs.

## How a number gets computed

Follow the data — it flows in one direction:

0. **Load the data** — `pdg_loader.py:load_pdg_data`. Pull the experimental masses
   and reference widths straight from the PDG SQLite dump (see the PDG data section
   below). This feeds the fit and the benchmarks.

1. **Fit** — `fitter.py:get_or_fit_parameters`. Either load cached
   `[alpha_s, b, c, sigma]` from `results/<sector>/params.csv`, or, if there's no
   cache, run `scipy.optimize.least_squares` (method `trf` — trust-region
   reflective, which handles bounds and the under-determined B_c sector that plain
   Levenberg–Marquardt can't). The objective is a **true dimensionless chi-square**:
   each residual is `(model − experiment) / sigma_physical`, not a hand-tuned
   weight (see the chi-square section). Both masses *and* the measured ground/first-
   excited leptonic `e+e-` widths go into the objective, so the wavefunction at the
   origin helps pin the potential — not masses alone.

2. **Solve** — `gem_solver.py:solve_gem`. This is the Gaussian Expansion Method.
   Lay down 25 Gaussians with widths in geometric progression, build the
   Hamiltonian `H` and overlap `S` from closed-form Gamma-function integrals — no
   numerical quadrature, the integrals are analytic, that's the trick — and solve
   the generalized eigenproblem `H c = E S c` with `scipy.linalg.eigh`. Out come
   energies, wavefunctions on a grid, the coefficients `c`, and the widths
   `nu_array`.

3. **Correct** — `observables.py`. The GEM gives you the leading spectrum; fine
   structure is perturbation theory on top, computed as expectation values over the
   eigenvectors:
   - `calc_so_shift_exact` — spin-orbit (LS), L > 0 only
   - `calc_tensor_shift_exact` — diagonal tensor splitting, triplets
   - `calc_tensor_mixing_exact` — the off-diagonal `<2^3S_1 | V_T | 1^3D_1>` that
     mixes S and D
   - `get_wfo_exact` — `|R(0)|^2` via the hypervirial theorem. You never evaluate
     the wavefunction at `r=0` directly; the theorem hands it to you as an
     expectation value, which is far more stable. This matters because every
     S-wave decay width leans on it.

4. **Decay** — `decay_models.py`. Feed `|R(0)|^2` and the masses into the width
   formulas: leptonic `V -> e+e-`, two-photon `P -> γγ`, E1/M1 radiative
   transitions, and 3P0 hadronic pair creation (e.g. `ψ(3770) -> D D̄`).

5. **Report** — `metrics.py` + `generate_report.py`. Tabulate, export CSVs, compute
   chi-square, write the consolidated report and `REPORT.md`.

## Things worth knowing

**`QuarkoniumSystem`** (`gem_solver.py`) is the parameter bag: quark masses `m_1` /
`m_2`, reduced mass `mu`, and the Cornell parameters `alpha_s`, `b` (string tension,
GeV²), `c` (zero-point shift, GeV). The hyperfine smearing `sigma_smear` defaults to
`sqrt(mu)` when you don't pass it — it's tied to the mass scale on purpose, so don't
hardcode it.

**State names are the keys.** Format: `"(n^{2S+1}L_J) symbol"`, e.g. `"(1^3S) J/ψ"`,
`"(1^3P_2) χ_c"`, `"(1^3D_1) ψ"`. These strings are dictionary keys all over
`run_spectrum.py` — change one and the lookups break silently.

**Units.** GeV everywhere internally. MeV only when reporting mass errors. Widths in
keV.

**Sectors.** `bb` bottomonium, `cc` charmonium, `bc` the B_c meson, `cu` the D meson.
The B_c is the special case: its `alpha_s` and `sigma` are *not* fitted — there
aren't enough known states to pin them down. Instead they're interpolated
logarithmically (running-coupling style) between the fitted `cc` and `bb` values at
the B_c reduced-mass scale, and their uncertainties are propagated through that
interpolation.

## Where the experimental data comes from

All experimental numbers — masses and their PDG errors for the fit, plus the `e+e-`
and `γγ` reference widths — are read live from the official PDG SQLite dump,
`data/pdg-*.sqlite`. There used to be a hand-edited `data/pdg_data.json`; it's gone.
`pdg_loader.py:load_pdg_data()` reconstructs the exact same dict shape the JSON had,
so `run_spectrum.py`, `metrics.py`, and `test_spectra.py` just call it and don't care
where the numbers originated. The loader auto-discovers the newest `data/pdg*.sqlite`,
so dropping in a newer PDG edition is a file swap — no code change.

The one thing that *can't* be read from the database is which measured particle is
which `(n^{2S+1}L_J)` state — that's physics, not data. It lives in `STATE_MAP` in
`pdg_loader.py`, keyed by PDG identifier (e.g. `M070` = J/ψ(1S), `M049` = Υ(1S)). If
you add a state, add its PDG id there. `None` means "not observed yet" (the two B_c
triplet states) and is emitted as null, which the fitter skips.

Two gotchas worth knowing:

- **Widths come in two flavors in the DB.** `e+e-` is stored as a direct partial
  width in keV; `γγ` is sometimes a direct width, sometimes only a branching
  fraction you must multiply by the total width. `pdg_loader.py` keys off the unit to
  tell them apart — don't assume a decay node is always one or the other.
- **Selection rules are applied on purpose.** `e+e-` widths are only emitted for
  vector (`J^PC = 1^--`) states and `γγ` only for `C = +`, `J ≠ 1` states. That's not
  the model's job — it's so the benchmark doesn't compare a width the DB happens to
  list (e.g. χ_c1 → e+e-) against a process the decay model doesn't compute.

## Uncertainties and chi-square — read this, it's subtle

There are **two different uncertainties** in this project and conflating them was the
original sin of the old code:

1. **The `±` on a value** is the *model* uncertainty: finite-difference propagation
   of the fitted Cornell covariance (`propagate_uncertainty` /
   `propagate_transition_uncertainty` in `run_spectrum.py`). It answers "how much does
   this prediction move when the fitted parameters wiggle within their errors." B_c
   inherits its errors through the log-interpolation.

2. **The `sigma` in a chi-square or a pull** is the *physical* error bar:
   `sigma = quad(sigma_exp, sigma_theory)`. This is the one that matters for "is the
   fit any good," and it is **not** the model covariance.

Why the distinction matters: PDG experimental mass errors are tiny — 0.1–1 MeV (see
`pdg_loader`'s `*_mass_err_GeV`). A potential model cannot reach that; its honest
accuracy is a **theory systematic**, `SIGMA_THEORY_MEV = 10` MeV on masses and
`WIDTH_THEORY_FRAC = 0.30` (30%) on the NR width formulae (both in `fitter.py`).
Those represent neglected physics — relativistic `O(v^2/c^2)`, coupled channels,
quenching. So:

- The fit objective and the goodness-of-fit chi-square both divide residuals by this
  physical `sigma`. That makes `chi^2/dof ≈ 1` mean "model reproduces the data to
  within its intrinsic systematic." Bottomonium lands near 1; charmonium lands above
  it (charm is lighter, more relativistic) — that's real physics, not a bad fit.
- The **RMS mass deviation** is the model-independent headline number; it assumes
  nothing about `sigma`. Quote it first.
- The old code instead divided by the inflated *model* covariance (~45 MeV), which
  forced `chi^2/dof << 1` and made everything look artificially perfect. Don't
  reintroduce that.

`dof = N_states − N_free_params`. The B_c (2 masses, 2 free) and D (1 mass, 1 free)
sectors have `dof ≤ 0` — exactly determined, so a reduced chi-square is *undefined*
and reported as `—`, never as a negative number.

**Overfitting** is checked honestly by `cross_validate.py`, not by staring at
`chi^2/dof`: fit on half the levels, measure RMS on the held-out half, swap. Test-RMS
≈ train-RMS ⇒ the rigid 4-parameter Cornell form generalizes.

Sanity check: `observables.check_virial_theorem` should return ~1.0 for a converged
basis (it includes the hyperfine term; pass `spin` for S-waves). If it drifts from 1,
the basis isn't converged and every downstream number is suspect.

## Output (`results/`)

The layout is **per-sector subfolders** plus a `summary/` for global artifacts.
`paths.py` is the single source of truth — writers and readers both go through it, so
filenames never drift. Don't hand-build `results/...` paths; call `paths.params_csv`,
`paths.errors_csv`, `paths.summary_csv`, etc.

```text
results/
  <sector>/            bb, cc, bc, cu
    params.csv         cached Cornell params + errors + chi2_fit, dof, chi2_per_dof
    errors.csv         mass vs PDG per state, with Mass_Err_GeV and (physical) Pull_sigma
    observables.csv    leptonic / two-photon widths per state, with Error_keV
    <Wave>_Wave_GEM_Coefficients.csv   raw Gaussian basis coefficients
  summary/
    goodness_of_fit.csv     per-sector + global chi², dof, chi²/dof, RMS deviation
    consolidated_report.csv everything — Uncertainty, Pull_sigma, Chi2_contrib
    radiative_decays.csv    E1/M1 transition widths
    cross_validation.csv    train/test RMS per sector (from cross_validate.py)
```

`generate_report.py` reads these and writes `REPORT.md`. `results/` is gitignored
regenerated output — safe to delete, the pipeline rebuilds it. The one thing you
can't cheaply rebuild is `<sector>/params.csv`: deleting those forces the slow refit.
