# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses a **conda environment named `400`**. Always run Python via:

```bash
conda run -n 400 python scripts/run_spectrum.py
```

Never use bare `python` or `python3`. Dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`.

## Common Commands

**Run the full spectroscopy pipeline** (fits parameters if no cached CSV exists, then generates masses, observables, and decay widths for all sectors):
```bash
conda run -n 400 python scripts/run_spectrum.py
```

**Run benchmarks against PDG data** (requires `run_spectrum.py` to have been run first to generate `results/*_errors.csv`):
```bash
conda run -n 400 python scripts/test_spectra.py
```

**Force parameter refit for a sector**: delete the corresponding `results/<sector>_params.csv` file (e.g. `results/bb_params.csv`) before running `run_spectrum.py`. The fitter caches results to avoid re-running slow optimizations.

**Legacy QHO validation**:
```bash
conda run -n 400 python qho/plot_qho.py
```

## Architecture

### Physics Pipeline

The computation flows through these stages:

1. **`fitter.py` → `get_or_fit_parameters()`**: Loads cached Cornell potential parameters `[alpha_s, b, c]` from `results/<sector>_params.csv`, or runs Levenberg-Marquardt (`scipy.optimize.least_squares`, TRF method) against PDG masses in `data/pdg_data.json` if missing.

2. **`gem_solver.py` → `solve_gem()`**: Builds the exact Hamiltonian and overlap matrices using analytical Gamma-function integrals over a basis of N=25 Gaussians with widths in geometric progression. Solves the generalized eigenvalue problem `H c = E S c` via `scipy.linalg.eigh`. Returns eigenvalues, spatial wavefunctions on a grid, eigenvectors (GEM coefficients), and the `nu_array` (Gaussian widths).

3. **`observables.py`**: Computes perturbative relativistic corrections as expectation values over GEM eigenvectors:
   - `calc_so_shift_exact()` — Spin-orbit (LS) coupling for L > 0 states
   - `calc_tensor_shift_exact()` — Diagonal tensor shift for triplet states
   - `calc_tensor_mixing_exact()` — Off-diagonal `<2^3S_1 | V_T | 1^3D_1>` element for S-D mixing
   - `get_wfo_exact()` / hypervirial theorem — Wavefunction at the origin `|R(0)|^2` without direct evaluation at r=0

4. **`decay_models.py`**: Computes decay widths using `|R(0)|^2` from the hypervirial theorem:
   - Leptonic (`Γ(V → e+e-)`)
   - Two-photon (`Γ(P → γγ)`)
   - Radiative E1 and M1 transitions (photon emission between states)
   - Hadronic 3P0 vacuum pair creation (e.g. `ψ(3770) → D + D̄`)

5. **`metrics.py`**: Formats tabulated output, exports GEM coefficients and observables to CSV, and generates the consolidated error report at `results/consolidated_error_report.csv`.

### Key Data Classes and Conventions

**`QuarkoniumSystem`** (in `gem_solver.py`): Central parameter object holding quark masses `m_1`, `m_2`, reduced mass `mu`, and Cornell potential parameters `alpha_s`, `b` (string tension, GeV²), `c` (zero-point energy shift, GeV). The smearing parameter `sigma_smear` for the hyperfine contact interaction is automatically tied to `sqrt(mu)`.

**State naming convention**: `"(n^{2S+1}L_J) symbol"` — e.g. `"(1^3S) J/ψ"`, `"(1^3P_2) χ_c"`, `"(1^3D_1) ψ"`. State names are used as dictionary keys throughout `run_spectrum.py`.

**Units**: Masses in GeV internally; converted to MeV for error reporting. Decay widths in keV.

**Sectors**: `bb` (Bottomonium), `cc` (Charmonium), `bc` (B_c meson), `cu` (D meson). The B_c `alpha_s` is not fitted freely — it is fixed by QFT logarithmic interpolation between the fitted `cc` and `bb` values using the reduced mass scale.

### Output Files (`results/`)

| File | Contents |
|------|----------|
| `<sector>_params.csv` | Cached Cornell potential parameters `[alpha_s, b, c]` with errors |
| `<sector>_observables.csv` | Leptonic/two-photon widths per state |
| `<sector>_errors.csv` | Mass comparison vs PDG per state |
| `<sector>_<Wave>_GEM_Coefficients.csv` | Raw Gaussian basis coefficients |
| `consolidated_error_report.csv` | Global accuracy summary across all sectors |
| `radiative_decays.csv` | E1/M1 transition widths |
