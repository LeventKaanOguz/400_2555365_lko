

# PHYS400 Project: Variational Approach to the Energy Spectra of Heavy Mesons

> **DEPRECATION NOTICE:** This repository is deprecated. Please refer to the successor repository on GitHub: [LeventKaanOguz/Quarkonium-System](https://github.com/LeventKaanOguz/Quarkonium-System) for the latest code, documentation, and updates.

## Abstract

Quantum Chromodynamics (QCD) is the fundamental theory describing the strong interactions between quarks and gluons. While highly successful at high energies (where asymptotic freedom allows for perturbative calculations), QCD becomes strongly coupled and non-perturbative at low energies, making direct analytical calculations of bound states like protons and mesons exceptionally difficult.

Heavy mesons, composed of massive charm ($c$) and bottom ($b$) quarks, provide a unique window into this non-perturbative regime. Due to their large masses ($m_c \approx 1.5$ GeV, $m_b \approx 4.7$ GeV), these quarks move at non-relativistic velocities within the bound state ($v^2/c^2 \ll 1$). This justifies treating the meson as a non-relativistic quantum mechanical two-body problem, governed by the Schrödinger equation with an effective QCD-inspired potential.

This project implements a high-precision, analytically resolved variational framework to compute the mass spectra, wavefunctions, and physical observables (decay widths, transition rates) for Heavy Quarkonia ($c\bar{c}$ Charmonium, $b\bar{b}$ Bottomonium), the $B_c$ meson ($b\bar{c}$), and the $D$ meson ($c\bar{u}$). By leveraging the **Gaussian Expansion Method (GEM)**, the computational pipeline sidesteps numerical singularities inherent in grid-based solvers, achieving sub-1% agreement with experimental Particle Data Group (PDG) measurements.

---

## Theoretical Framework

### The Cornell Potential

The strong interaction between a quark and an antiquark is effectively modeled by the Cornell potential, which linearly combines two critical behaviors:

$$
V(r) = -\frac{4}{3}\frac{\alpha_s}{r} + \sigma r + c
$$

1. **Short-Distance Coulombic Regime ($-\frac{1}{r}$):** Governs the interaction at very small separations, mediated by the exchange of a single massless gluon. The prefactor $4/3$ is the color factor for a meson (a color-singlet state), and $\alpha_s$ is the strong coupling constant.
2. **Long-Distance Confinement ($\sigma r$):** As quarks are pulled apart, gluon self-interactions form a "flux tube" or string. The energy stored in this string grows linearly with distance (where $\sigma$, or string tension, is $\approx 0.18$ GeV$^2$). This prevents isolated quarks from existing, leading to color confinement.
3. **Mass Shift ($c$):** A constant zero-point energy shift to calibrate the bare quark masses ($M_{\text{bare}} = m_1 + m_2$) against the physical state masses.

### Relativistic Fine-Structure Corrections

The pure Cornell potential only predicts degenerate "bare" energy states. To resolve the physical fine and hyperfine splitting (such as the 114 MeV gap between the singlet $\eta_c$ and the triplet $J/\psi$), we compute order $O(\alpha_s^2 / m^2)$ relativistic Breit-Fermi corrections as expectation values $\langle \psi | \Delta H | \psi \rangle$.

- **Spin-Spin (Hyperfine) Interaction:**
  Affects primarily S-wave ($L=0$) states. It arises from the magnetic dipole interaction between quark spins.

  $$
  \Delta E_{HF} = \frac{32 \pi \alpha_s}{9 m_1 m_2} \langle \delta^3(\vec{r}) \rangle \langle \vec{S}_1 \cdot \vec{S}_2 \rangle
  $$

  *Implementation detail:* The Dirac delta function $\delta^3(\vec{r})$ is analytically unstable. We systematically "smear" it using a Gaussian distribution proportional to the reduced mass, providing a mathematically robust and physically accurate interaction core.
- **Spin-Orbit (LS) Coupling:**
  For P-wave and D-wave states ($L > 0, S > 0$), the interaction between the quarks' orbital angular momentum and their spins splits the states (e.g., resolving the $\chi_{c0}, \chi_{c1}, \chi_{c2}$ triplet). The interaction includes both a positive Coulombic vector term and a negative scalar confinement (Thomas precession) term.
- **Tensor Interaction and S-D Mixing:**
  The tensor operator couples the quark spins to their spatial orientation. Crucially, it mixes states with different orbital angular momenta but the same total spin and $J$, such as the $1^3S_1$ and $1^3D_1$ states. The matrix element $\langle ^3S_1 | V_T | ^3D_1 \rangle$ is explicitly constructed and diagonalized to find the true physical masses for the $J/\psi$ and $\psi(3770)$.

---

## Methodology & Computational Highlights

### 1. The Gaussian Expansion Method (GEM)

Traditional finite-difference (FD) methods struggle with the $1/r$ divergence at the origin and demand massive grid sizes. Instead, we expand the radial wavefunction $u(r)$ using a basis of $N=25$ non-orthogonal Gaussian functions:

$$
u_{nl}(r) = \sum_{i=1}^{N} c_i \cdot r^{l+1} e^{-\nu_i r^2}
$$

The widths $\nu_i$ are distributed in a **geometric progression** to efficiently cover both the ultra-compact core ($r \to 0$) and the diffuse asymptotic tail ($r \to \infty$).

### 2. Exact Analytical Integrals

Because Gaussians possess well-defined analytical properties, all matrix elements (Overlap $\mathbf{S}$, Kinetic $\mathbf{T}$, Coulombic, Linear, and Centrifugal potentials) are evaluated **exactly** using Gamma functions:

$$
\int_0^\infty r^p e^{-\nu r^2} dr = \frac{1}{2} \frac{\Gamma\left(\frac{p+1}{2}\right)}{\nu^{(p+1)/2}}
$$

This eliminates numerical integration error completely. Solving the system then reduces to a Generalized Eigenvalue Problem ($\mathbf{H}\vec{c} = E\mathbf{S}\vec{c}$), solved via `scipy.linalg.eigh`.

### 3. The Hypervirial Theorem & Decay Models

To compute decay constants ($f_{B_c}$) and leptonic widths (e.g., $J/\psi \to e^+ e^-$), the probability density at the origin $|R(0)|^2$ must be precisely known. Since a finite Gaussian basis inherently smooths the physical $r=0$ "cusp", evaluating the expansion directly at zero introduces systematic underestimation.
Instead, we apply the Schwinger/Hypervirial relation:

$$
|R(0)|^2 = 2 \mu \left\langle \frac{dV}{dr} \right\rangle
$$

This projects the contact probability onto a global expectation value, yielding highly accurate, cusp-independent calculations.

Using $|R(0)|^2$, we can compute physical decay rates, such as the **Leptonic Decay** ($V \rightarrow e^+e^-$) for Vector Mesons (Spin = 1) like $J/\psi$ or $\Upsilon(1S)$:

$$
\Gamma_{e^+e^-} = \frac{4 \alpha_{em}^2 e_q^2}{M^2} |R(0)|^2 \left( 1 - \frac{16\alpha_s}{3\pi} \right)
$$

where $\alpha_{em} \approx 1/137.036$, $e_q$ is the fractional quark charge, and the rightmost term provides the first-order perturbative QCD correction. To properly account for hard gluon exchanges during annihilation at very short ranges, we evaluate the high-energy running coupling $\alpha_s(m_q)$ at the heavy quark mass scale ($\approx 0.35$ for charmonium, $\approx 0.20$ for bottomonium) rather than using the long-range parameter fitted to the Cornell potential.

Beyond leptonic and two-photon annihilation decays, the framework evaluates:

- **Radiative Transitions (E1 & M1):** Transitions between states emitting a photon (e.g., $J/\psi \to \eta_c + \gamma$) using analytically resolved spatial overlap integrals and dipole approximations.
- **Hadronic Decays (3P0 Vacuum Pair Creation Model):** Phenomenological modeling of strong decays (e.g., $\psi(3770) \to D + \bar{D}$), fitting the vacuum pair-creation parameter $\gamma$ to match experimental widths.

### 4. Automated Global Fitter & Error Metrics

The framework features an automated optimization routine (`fitter.py`) using Levenberg-Marquardt (Trust Region Reflective) least-squares regression. It dynamically fine-tunes the physical parameters ($m, \alpha_s, b, c$) against a weighted subset of Particle Data Group empirical masses, effectively allowing the model to "learn" the optimal QCD potential parameters.

To ensure robustness, theoretical outputs are automatically benchmarked against empirical data (`data/pdg_data.json`). The consolidated error report calculates the **Absolute Error (MeV)**, **Percentage Error (%)**, and **Mean Squared Error (MSE)** for every generated mass and decay observable across all evaluated sectors.

---

## Repository Structure

- **`src/quarkonia/`**: Core Python package containing the physics logic.
  - `gem_solver.py`: The analytical engine. Constructs the exact Hamiltonian/Overlap matrices using Gamma functions and solves the generalized eigenvalue problem.
  - `observables.py`: Computes physical expectations, hypervirial amplitudes, relativistic perturbative shifts (Spin-Orbit/Tensor), and state mixing.
  - `fitter.py`: Executes the multi-parameter $\chi^2$ global fit against the PDG data using `scipy.optimize.least_squares`.
  - `metrics.py`: Analyzes the calculated eigenspectrum, formatting output tables and exporting the optimized GEM coefficients to CSV.
- **`scripts/`**: Executable scripts.
  - `run_spectrum.py`: The master script. It dynamically handles the $b\bar{b}$, $c\bar{c}$, and $b\bar{c}$ configurations, triggering the fitter, generating the full quantum multiplet up to $n=3, L=2$, computing decay observables, and formatting the output.
- **`data/`**:
  - `pdg_data.json`: Static, digitized experimental mass values from the Particle Data Group, acting as the ground-truth benchmark.
- **`results/`**: Outputs generated by the code.
  - Parameter CSVs (`bb_params.csv`, `cc_params.csv`, etc.).
  - Eigenspectrum summaries and individual observable reports (`bb_observables.csv`, etc.).
  - `radiative_decays.csv`: Comprehensive mapping of E1/M1 transitions.
  - `consolidated_error_report.csv`: Global accuracy analysis evaluating Abs Error, % Error, and MSE against PDG data.
  - Wavefunction expansion parameters (GEM Coefficients).
- **`qho/`** (Legacy):
  - Contains preliminary validation scripts solving the Quantum Harmonic Oscillator with quartic perturbations to verify the stability of the Gaussian Expansion Method versus Finite Differences.

---

## Setup and Installation

It is highly recommended to use a Python virtual environment to prevent dependency conflicts.

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # venv\Scripts\activate   # On Windows
   ```
3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *Required packages primarily include `numpy`, `scipy`, `pandas`, and `matplotlib`.*

---

## Running the Code

### Generative Spectroscopy Pipeline

To compute the full spectrum of heavy mesons (masses, errors, decay observables), execute the main spectrum runner:

```bash
python scripts/run_spectrum.py
```

**What it does:**

1. Loads the PDG experimental data.
2. For each sector ($b\bar{b}$, $c\bar{c}$, $b\bar{c}$, $c\bar{u}$), checks if optimized parameters exist in `results/`. If not, it runs the global optimization fitter.
3. Generates the full basis of $S, P,$ and $D$ wave states using GEM, alongside spatial RMS radii.
4. Computes relativistic perturbative corrections (Spin-Orbit, Tensor).
5. Mixes the $1^3S_1$ and $1^3D_1$ states.
6. Computes decay rates (Leptonic, Two-Photon, Radiative E1/M1, and Hadronic 3P0 widths).
7. Outputs detailed, tabulated comparison charts directly to the console and aggregates a comprehensive `consolidated_error_report.csv` inside `results/`.

### Legacy Validation: Quantum Harmonic Oscillator

To verify the core variational mathematics, you can run the legacy comparative analysis of a perturbed QHO:

```bash
python qho/plot_qho.py
```

This will:

- Solve the oscillator numerically via finite differences.
- Solve the oscillator variationally using various trial functions (including Gaussians).
- Compare absolute errors, saving a high-resolution composite figure to `results/figures/qho_showcase.png` and results table to `results/tables/qho_numerical_results.csv`.
