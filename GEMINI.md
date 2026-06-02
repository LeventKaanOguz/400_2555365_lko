# Heavy Quarkonium Spectroscopy Project (GEM)

## Project Overview
This project implements a high-precision variational framework to compute the mass spectra, wavefunctions, and physical observables of Heavy Quarkonia ($c\bar{c}$, $b\bar{b}$), the $B_c$ meson, and the $D$ meson. It uses the **Gaussian Expansion Method (GEM)** to solve the non-relativistic Schrödinger equation with a QCD-inspired Cornell potential and relativistic Breit-Fermi corrections.

### Main Technologies
- **Python 3**: Core implementation language.
- **NumPy & SciPy**: Numerical linear algebra and optimization (e.g., `scipy.linalg.eigh` for the eigenvalue problem, `scipy.optimize.least_squares` for parameter fitting).
- **Pandas**: Data handling and CSV export for results.
- **Matplotlib**: Visualization of wavefunctions and spectra.

### Architecture
- `src/quarkonia/`: Core physics package.
  - `gem_solver.py`: Analytical engine using Gamma-function integrals for exact matrix elements.
  - `observables.py`: Computes relativistic shifts (Spin-Orbit, Tensor), state mixing, and expectation values.
  - `decay_models.py`: Calculates leptonic, two-photon, radiative, and hadronic decay widths.
  - `fitter.py`: Automated parameter optimization against PDG data.
  - `metrics.py`: Formats outputs and generates error reports.
- `scripts/`:
  - `run_spectrum.py`: Master runner for the full pipeline.
  - `test_spectra.py`: Benchmarking script for theoretical vs. experimental data.
- `data/`: Contains `pdg_data.json` with ground-truth experimental values.
- `results/`: Output directory for optimized parameters, observables, and error reports.

---

## Building and Running

### Environment Setup
A conda environment named `400` is the standard for this project.
```bash
# Install dependencies
pip install -r requirements.txt
```

### Key Commands
- **Run Full Pipeline**: Generates masses, observables, and decay widths for all sectors. Fits parameters if cached CSVs are missing in `results/`.
  ```bash
  conda run -n 400 python scripts/run_spectrum.py
  ```
- **Run Benchmarks**: Compares calculated results against PDG data (must run `run_spectrum.py` first).
  ```bash
  conda run -n 400 python scripts/test_spectra.py
  ```
- **Legacy QHO Validation**: Verifies the GEM engine against a perturbed Quantum Harmonic Oscillator.
  ```bash
  conda run -n 400 python qho/plot_qho.py
  ```

---

## Development Conventions

### Physics & Units
- **Internal Units**: Masses and energies are in **GeV**; distance in **fm** (or implicitly via GeV inverse).
- **Output Units**: Masses in MeV (for errors), decay widths in **keV**.
- **Fundamental Constants**: Reduced Planck constant $\hbar = 1.0$ (natural units) is the default.

### Coding Patterns
- **Parameter Caching**: Optimal Cornell potential parameters are cached in `results/<sector>_params.csv`. Delete these files to force a refit.
- **Data Classes**: `QuarkoniumSystem` in `gem_solver.py` is the primary container for physical parameters.
- **State Naming**: Follows the convention `(n^{2S+1}L_J) symbol` (e.g., `(1^3S) J/ψ`). These strings are used as keys for lookups.
- **Hypervirial Theorem**: Always use the hypervirial relation (Schwinger relation) for wavefunction-at-origin $|R(0)|^2$ calculations to avoid numerical cusp instabilities.

### Testing & Validation
- New physics features should be validated against PDG data using `test_spectra.py`.
- The `results/consolidated_error_report.csv` is the primary metric for global model accuracy.
