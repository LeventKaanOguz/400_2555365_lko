import numpy as np
import os
import sys
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.numerical import solve_fd, solve_shooting
from scripts.perturbation import calc_numerical_pert_energy
from scripts.variational import normalize_ansatz, compute_expectation_value, solve_gem

# --- System Parameters ---
HBAR = 1.0
M_Q = 4.730
MU = M_Q / 2.0
ALPHA_S = 0.232  # Strong coupling constant for the Coulomb-like interaction
B = 0.192  # Linear confinement string tension
C = -0.0796  # Constant potential shift
SIGMA_SMEAR = 0.005  # Smearing parameter for spin-spin interaction (GeV)
EXP_MASS_1S = 9.3987  # eta_b (1S) mass in GeV
EXP_MASS_1S_TRIPLET = 9.4603  # Upsilon (1S) mass in GeV
EXP_MASS_2S = 10.0234  # Upsilon (2S) mass in GeV


def v_cornell_3d(r_val):
    return -(4.0 / 3.0) * ALPHA_S / r_val + B * r_val + C


def v_cornell_3d_smeared(r_val, spin=0):
    v_base = v_cornell_3d(r_val)
    spin_dot = -0.75 if spin == 0 else 0.25
    hf_coeff = (32.0 * np.pi * ALPHA_S) / (9.0 * M_Q**2)
    smeared_delta = (SIGMA_SMEAR / np.sqrt(np.pi)) ** 3 * np.exp(
        -(SIGMA_SMEAR**2) * r_val**2
    )
    return v_base + hf_coeff * spin_dot * smeared_delta


def run_comparisons():
    L = 15.0 / max(0.5, B)
    N = 6000
    r = np.linspace(L / N, L, N)
    dr = r[1] - r[0]

    print("=" * 85)
    print(" " * 26 + "3D CORNELL POTENTIAL ANALYSIS")
    print("=" * 85)
    print("\n--- System & Potential Parameters ---")
    print(f"Potential Model          : V(r) = -(4/3)*alpha_s/r + b*r + c")
    print(f"Quark Mass (m_q)         : {M_Q:.3f} GeV")
    print(f"Reduced Mass (mu)        : {MU:.3f} GeV")
    print(f"Strong Coupling (alpha_s): {ALPHA_S:.3f}")
    print(f"String Tension (b)       : {B:.3f} GeV^2")
    print(f"Potential Shift (c)      : {C:.3f} GeV")
    print(f"Planck Constant (hbar)   : {HBAR:.1f}")
    print(
        f"Grid Setup               : {N} points from r={r[0]:.4f} to r={r[-1]:.4f} (dr={dr:.4f})\n"
    )

    print("--- Experimental Reference Masses ---")
    print(f"eta_b (1S, S=0)          : {EXP_MASS_1S:.4f} GeV")
    print(f"Upsilon (1S, S=1)        : {EXP_MASS_1S_TRIPLET:.4f} GeV")
    print(f"Upsilon (2S, S=1)        : {EXP_MASS_2S:.4f} GeV\n")

    print("--- Methodology Details ---")
    print(
        "1. Perturbation Theory (Numerical): Treats the linear confinement term (b*r) as a"
    )
    print(
        "   perturbation over the Coulombic potential. Calculates up to 2nd-order corrections."
    )
    print(
        "2. Variational Method             : Optimizes trial wavefunctions (Harmonic, Hydrogenic,"
    )
    print("   Power) non-perturbatively using a smeared Gaussian spin-spin potential.")
    print(
        "3. Gaussian Expansion Method (GEM): Solves the Generalized Eigenvalue Problem using 25"
    )
    print("   non-orthogonal Gaussian basis functions distributed geometrically.")
    print(
        "4. Numerical Matrix FD            : Discretizes the Hamiltonian on a 1D radial grid"
    )
    print(
        "   using Finite Differences and diagonalizes the resulting sparse tridiagonal matrix."
    )
    print(
        "5. Shooting Method (IVP)          : Uses root-finding with scipy.integrate.solve_ivp"
    )
    print("   to meet the wavefunction boundary conditions at infinity.\n")

    print(
        "Note: Spin-spin hyperfine splitting is applied perturbatively for most methods, but"
    )
    print(
        "non-perturbatively (smeared) for the Variational Method. S=0 is used for eta_b (1S),"
    )
    print("and S=1 for Upsilon (1S and 2S) states.\n")

    v_total = v_cornell_3d(r)

    def calc_hf_shift(u_array, spin=0):
        # 2nd order forward difference for u'(0) since u(0) = 0
        u_prime_0 = (4.0 * u_array[0] - u_array[1]) / (2.0 * dr)
        factor = -(2.0 / 3.0) if spin == 0 else (2.0 / 9.0)
        return factor * (ALPHA_S / M_Q**2) * (u_prime_0**2)

    # ==========================================
    # 1. PERTURBATION THEORY (NUMERICAL)
    # ==========================================
    # Perturb around the Coulomb-like potential with the linear term
    v_unperturbed = -(4.0 / 3.0) * ALPHA_S / r + C
    v_pert = B * r

    evals_unperturbed, u_unperturbed = solve_fd(r, v_unperturbed, HBAR, MU)

    hf_shift_1s_pert = calc_hf_shift(u_unperturbed[:, 0], spin=0)
    hf_shift_2s_pert = calc_hf_shift(u_unperturbed[:, 1], spin=1)

    e1s_pert_1st, e1s_pert_2nd = calc_numerical_pert_energy(
        0, r, v_pert, evals_unperturbed, u_unperturbed
    )
    e2s_pert_1st, e2s_pert_2nd = calc_numerical_pert_energy(
        1, r, v_pert, evals_unperturbed, u_unperturbed
    )

    e1s_pert_1st += hf_shift_1s_pert
    e1s_pert_2nd += hf_shift_1s_pert
    e2s_pert_1st += hf_shift_2s_pert
    e2s_pert_2nd += hf_shift_2s_pert

    # ==========================================
    # 2. VARIATIONAL METHOD (Radial Ansatz)
    # ==========================================
    # Reduced radial function u(r) = r * R(r)
    v_smeared_spin0 = v_cornell_3d_smeared(r, spin=0)
    v_smeared_spin1 = v_cornell_3d_smeared(r, spin=1)

    def ansatz_1s(r_val, beta):
        return r_val * np.exp(-0.5 * beta**2 * r_val**2)

    def ansatz_2s(r_val, beta):
        return r_val * (r_val**2 - 1.5 / beta**2) * np.exp(-0.5 * beta**2 * r_val**2)

    def ansatz_1s_hyd(r_val, beta):
        # Hydrogenic 1s (n=1, l=0): L^(1)_0(x) = 1
        return r_val * np.exp(-beta * r_val)

    def ansatz_2s_hyd(r_val, beta):
        # Hydrogenic 2s (n=2, l=0): L^(1)_1(x) = 2 - x
        return r_val * (2.0 - beta * r_val) * np.exp(-0.5 * beta * r_val)

    def ansatz_1s_pow(r_val, r_n):
        return r_val * (r_val**0) * np.exp(-((r_val / r_n) ** 2))

    def ansatz_2s_pow(r_val, r_n):
        return r_val * (r_val**1) * np.exp(-((r_val / r_n) ** 2))

    res1s = minimize(
        lambda p: compute_expectation_value(ansatz_1s, p, r, v_smeared_spin0, HBAR, MU),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_1s_opt = res1s.x[0]
    e1s_var = res1s.fun
    u1s_var = normalize_ansatz(ansatz_1s, [beta_1s_opt], r)

    res2s = minimize(
        lambda p: compute_expectation_value(ansatz_2s, p, r, v_smeared_spin1, HBAR, MU),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_2s_opt = res2s.x[0]
    e2s_var = res2s.fun
    u2s_var = normalize_ansatz(ansatz_2s, [beta_2s_opt], r)

    res1s_hyd = minimize(
        lambda p: compute_expectation_value(
            ansatz_1s_hyd, p, r, v_smeared_spin0, HBAR, MU
        ),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_1s_hyd_opt = res1s_hyd.x[0]
    e1s_var_hyd = res1s_hyd.fun
    u1s_var_hyd = normalize_ansatz(ansatz_1s_hyd, [beta_1s_hyd_opt], r)

    res2s_hyd = minimize(
        lambda p: compute_expectation_value(
            ansatz_2s_hyd, p, r, v_smeared_spin1, HBAR, MU
        ),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_2s_hyd_opt = res2s_hyd.x[0]
    e2s_var_hyd = res2s_hyd.fun
    u2s_var_hyd = normalize_ansatz(ansatz_2s_hyd, [beta_2s_hyd_opt], r)

    res1s_pow = minimize(
        lambda p: compute_expectation_value(
            ansatz_1s_pow, p, r, v_smeared_spin0, HBAR, MU
        ),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    rn_1s_opt = res1s_pow.x[0]
    e1s_var_pow = res1s_pow.fun
    u1s_var_pow = normalize_ansatz(ansatz_1s_pow, [rn_1s_opt], r)

    res2s_pow = minimize(
        lambda p: compute_expectation_value(
            ansatz_2s_pow, p, r, v_smeared_spin1, HBAR, MU
        ),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    rn_2s_opt = res2s_pow.x[0]
    e2s_var_pow = res2s_pow.fun
    u2s_var_pow = normalize_ansatz(ansatz_2s_pow, [rn_2s_opt], r)

    print("--- Variational Method Optimized Parameters ---")
    print(
        f"Harmonic Ansatz   | 1S beta: {beta_1s_opt:.6f} | 2S beta: {beta_2s_opt:.6f}"
    )
    print(
        f"Hydrogenic Ansatz | 1S beta: {beta_1s_hyd_opt:.6f} | 2S beta: {beta_2s_hyd_opt:.6f}"
    )
    print(f"Power Ansatz      | 1S r_n:  {rn_1s_opt:.6f} | 2S r_n:  {rn_2s_opt:.6f}\n")

    # ==========================================
    # 3. GAUSSIAN EXPANSION METHOD (GEM)
    # ==========================================
    n_basis_gem = 25
    r_min_gem = 0.03
    r_max_gem = 15.0
    l_gem = 0

    print("--- Gaussian Expansion Method (GEM) Parameters ---")
    print(f"Number of Basis Functions (n_basis) : {n_basis_gem}")
    print(f"Minimum Radial Limit (r_min)        : {r_min_gem}")
    print(f"Maximum Radial Limit (r_max)        : {r_max_gem}")
    print(f"Orbital Angular Momentum (l)        : {l_gem}\n")

    evals_gem, u_gem, evecs_gem, nu_gem = solve_gem(
        r,
        v_total,
        HBAR,
        MU,
        n_basis=n_basis_gem,
        r_min=r_min_gem,
        r_max=r_max_gem,
        l=l_gem,
    )
    e1s_gem_bare = evals_gem[0]
    e2s_gem_bare = evals_gem[1]
    u1s_gem = u_gem[:, 0]
    u2s_gem = u_gem[:, 1]

    e1s_gem = e1s_gem_bare + calc_hf_shift(u1s_gem, spin=0)
    e2s_gem = e2s_gem_bare + calc_hf_shift(u2s_gem, spin=1)

    print("--- GEM Optimized Parameters ---")
    print(
        f"{'Index':<7} | {'nu (width)':<15} | {'c_1S (Ground)':<18} | {'c_2S (1st Excited)'}"
    )
    print("-" * 67)
    for i in range(n_basis_gem):
        print(
            f"{i:<7} | {nu_gem[i]:<15.6e} | {evecs_gem[i, 0]:<18.6e} | {evecs_gem[i, 1]:.6e}"
        )
    print()

    # ==========================================
    # 4. NUMERICAL METHOD 1: FINITE DIFFERENCE
    # ==========================================
    evals_fd, u_fd = solve_fd(r, v_total, HBAR, MU)
    e1s_fd_bare = evals_fd[0]
    e2s_fd_bare = evals_fd[1]
    u1s_fd_arr = u_fd[:, 0]
    u2s_fd_arr = u_fd[:, 1]

    e1s_fd = e1s_fd_bare + calc_hf_shift(u1s_fd_arr, spin=0)
    e2s_fd = e2s_fd_bare + calc_hf_shift(u2s_fd_arr, spin=1)

    # ==========================================
    # 5. NUMERICAL METHOD 2: SHOOTING METHOD
    # ==========================================
    bracket_margin1s = max(0.1, abs(e1s_fd_bare) * 0.05)
    e1s_shoot_bare = solve_shooting(
        v_cornell_3d,
        e1s_fd_bare,
        None,
        L,
        HBAR,
        MU,
        bracket_margin1s,
        is_radial=True,
        x0=dr,
    )
    e1s_shoot = e1s_shoot_bare + calc_hf_shift(u1s_fd_arr, spin=0)

    bracket_margin2s = max(0.1, abs(e2s_fd_bare) * 0.05)
    e2s_shoot_bare = solve_shooting(
        v_cornell_3d,
        e2s_fd_bare,
        None,
        L,
        HBAR,
        MU,
        bracket_margin2s,
        is_radial=True,
        x0=dr,
    )
    e2s_shoot = e2s_shoot_bare + calc_hf_shift(u2s_fd_arr, spin=1)

    # ==========================================
    # SUMMARY TABLE
    # ==========================================
    print("--- Energy Eigenvalue Comparison Table ---")
    header = f"{'Calculation Method':<28} | {'E_1S (Ground State)':<20} | {'E_2S (1st Excited)'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(
        f"{'Perturbation Theory (1st)':<28} | {e1s_pert_1st:<20.6f} | {e2s_pert_1st:.6f}"
    )
    print(
        f"{'Perturbation Theory (2nd)':<28} | {e1s_pert_2nd:<20.6f} | {e2s_pert_2nd:.6f}"
    )
    print(f"{'Variational (Harmonic)':<28} | {e1s_var:<20.6f} | {e2s_var:.6f}")
    print(
        f"{'Variational (Hydrogenic)':<28} | {e1s_var_hyd:<20.6f} | {e2s_var_hyd:.6f}"
    )
    print(f"{'Variational (Power)':<28} | {e1s_var_pow:<20.6f} | {e2s_var_pow:.6f}")
    print(f"{'Variational (GEM)':<28} | {e1s_gem:<20.6f} | {e2s_gem:.6f}")
    print(f"{'Numerical: Matrix FD':<28} | {e1s_fd:<20.6f} | {e2s_fd:.6f}")
    print(f"{'Numerical: Shooting IVP':<28} | {e1s_shoot:<20.6f} | {e2s_shoot:.6f}")

    print("\n--- Physical Meson Mass Estimates (M = 2*m_q + E) ---")
    mass_header = f"{'Calculation Method':<28} | {'Mass 1S [GeV]':<15} | {'Diff [MeV]':<12} | {'% Error':<10} | {'Mass 2S [GeV]':<15} | {'Diff [MeV]':<12} | {'% Error':<10}"
    print("-" * len(mass_header))
    print(mass_header)
    print("-" * len(mass_header))

    methods_all = [
        "Perturbation (1st)",
        "Perturbation (2nd)",
        "Variational (Harmonic)",
        "Variational (Hydrogenic)",
        "Variational (Power)",
        "Variational (GEM)",
        "Shooting IVP",
        "Numerical: Matrix FD",
    ]
    energies_1s = [
        e1s_pert_1st,
        e1s_pert_2nd,
        e1s_var,
        e1s_var_hyd,
        e1s_var_pow,
        e1s_gem,
        e1s_shoot,
        e1s_fd,
    ]
    energies_2s = [
        e2s_pert_1st,
        e2s_pert_2nd,
        e2s_var,
        e2s_var_hyd,
        e2s_var_pow,
        e2s_gem,
        e2s_shoot,
        e2s_fd,
    ]

    masses_1s = [2 * M_Q + e for e in energies_1s]
    masses_2s = [2 * M_Q + e for e in energies_2s]
    mass_diffs_1s = [(m - EXP_MASS_1S) * 1000 for m in masses_1s]
    mass_diffs_2s = [(m - EXP_MASS_2S) * 1000 for m in masses_2s]
    mass_pct_errors_1s = [abs(m - EXP_MASS_1S) / EXP_MASS_1S * 100 for m in masses_1s]
    mass_pct_errors_2s = [abs(m - EXP_MASS_2S) / EXP_MASS_2S * 100 for m in masses_2s]

    for i, method in enumerate(methods_all):
        m1, d1 = masses_1s[i], mass_diffs_1s[i]
        m2, d2 = masses_2s[i], mass_diffs_2s[i]
        p1, p2 = mass_pct_errors_1s[i], mass_pct_errors_2s[i]
        print(
            f"{method:<28} | {m1:<15.4f} | {d1:<12.1f} | {p1:<10.3f} | {m2:<15.4f} | {d2:<12.1f} | {p2:<10.3f}"
        )
    print()

    print("--- Comparison with Literature (Akbar et al. 2024) ---")
    print(
        "Reference paper provided Experimental and Theoretical benchmarks for bottomonium."
    )

    # Include L=1 (P-wave) and L=2 (D-wave) centrifugal potential terms
    v_eff_1 = v_total + 1 * 2 * HBAR**2 / (2.0 * MU * r**2)
    evals_fd_1, u_fd_1 = solve_fd(r, v_eff_1, HBAR, MU)

    v_eff_2 = v_total + 2 * 3 * HBAR**2 / (2.0 * MU * r**2)
    evals_fd_2, u_fd_2 = solve_fd(r, v_eff_2, HBAR, MU)

    def get_mass(evals, u_arr, state_idx, spin, l):
        bare = evals[state_idx]
        # Hyperfine splitting is zero for L > 0 since wavefunctions vanish at the origin
        shift = calc_hf_shift(u_arr[:, state_idx], spin=spin) if l == 0 else 0.0
        return 2 * M_Q + bare + shift

    print(
        f"{'State':<13} | {'Our Work (FD)':<15} | {'Akbar Var Param':<15} | {'Akbar (2024)':<12} | {'Experimental':<18} | {'[27]':<8} | {'[33]':<8} | {'[34]':<8} | {'[35]':<8} | {'[25]':<8} | {'[36]':<4}"
    )
    print("-" * 147)
    table_data = [
        (
            "(1^1S) η_b",
            get_mass(evals_fd, u_fd, 0, 0, 0),
            "0.7828",
            "9.5535",
            "9.3987±0.002",
            "9.5615",
            "9.398",
            "9.398",
            "9.5079",
            "9.452",
            "-",
        ),
        (
            "(1^3S) Υ_b",
            get_mass(evals_fd, u_fd, 0, 1, 0),
            "0.7571",
            "9.5722",
            "9.4603±0.00026",
            "9.6478",
            "9.478",
            "9.460",
            "9.5229",
            "9.480",
            "-",
        ),
        (
            "(1^1P) h_b",
            get_mass(evals_fd_1, u_fd_1, 0, 0, 1),
            "0.5129",
            "9.9373",
            "9.8993±0.0008",
            "9.9324",
            "9.900",
            "9.894",
            "9.9279",
            "-",
            "-",
        ),
        (
            "(1^3P) χ",
            get_mass(evals_fd_1, u_fd_1, 0, 1, 1),
            "0.5096",
            "9.9391",
            "9.9122±0.0005",
            "9.9389",
            "9.912",
            "9.858",
            "9.9232",
            "-",
            "-",
        ),
        (
            "(1^1D) η_b2",
            get_mass(evals_fd_2, u_fd_2, 0, 0, 2),
            "0.4425",
            "10.1398",
            "-",
            "-",
            "10.163",
            "-",
            "10.1355",
            "-",
            "-",
        ),
        (
            "(1^3D) Υ",
            get_mass(evals_fd_2, u_fd_2, 0, 1, 2),
            "0.4422",
            "10.1399",
            "-",
            "-",
            "10.161",
            "-",
            "10.1548",
            "-",
            "-",
        ),
        (
            "(2^1S) η_b",
            get_mass(evals_fd, u_fd, 1, 0, 0),
            "0.62615",
            "9.9980",
            "-",
            "-",
            "9.990",
            "10.017",
            "10.0041",
            "10.030",
            "-",
        ),
        (
            "(2^3S) Υ",
            get_mass(evals_fd, u_fd, 1, 1, 0),
            "0.6215",
            "10.0052",
            "10.0233±0.0003",
            "10.0167",
            "10.023",
            "10.356",
            "10.0101",
            "10.055",
            "-",
        ),
        (
            "(2^1P) h_b",
            get_mass(evals_fd_1, u_fd_1, 1, 0, 1),
            "0.3924",
            "10.2210",
            "10.2598±0.0012",
            "10.2161",
            "10.260",
            "10.259",
            "-",
            "-",
            "-",
        ),
        (
            "(2^3P) χ",
            get_mass(evals_fd_1, u_fd_1, 1, 1, 1),
            "0.3909",
            "10.2288",
            "10.2687±0.0005",
            "-",
            "10.2232",
            "10.255",
            "-",
            "-",
            "-",
        ),
        (
            "(2^2D) η_b2",
            get_mass(evals_fd_2, u_fd_2, 1, 0, 2),
            "0.301959",
            "10.3780",
            "-",
            "-",
            "-",
            "10.450",
            "-",
            "-",
            "-",
        ),
        (
            "(2^3D) Υ",
            get_mass(evals_fd_2, u_fd_2, 1, 1, 2),
            "0.30170",
            "10.3783",
            "-",
            "-",
            "10.443",
            "10.442",
            "-",
            "-",
            "-",
        ),
    ]
    for row in table_data:
        print(
            f"{row[0]:<13} | {row[1]:<15.4f} | {row[2]:<15} | {row[3]:<12} | {row[4]:<18} | {row[5]:<8} | {row[6]:<8} | {row[7]:<8} | {row[8]:<8} | {row[9]:<8} | {row[10]:<4}"
        )
    print("=" * 147 + "\n")

    return {
        "r": r,
        "dr": dr,
        "v_total": v_total,
        "e1s_fd": e1s_fd,
        "e2s_fd": e2s_fd,
        "u1s_fd": u1s_fd_arr,
        "u2s_fd": u2s_fd_arr,
        "u1s_var": u1s_var,
        "u2s_var": u2s_var,
        "u1s_var_hyd": u1s_var_hyd,
        "u2s_var_hyd": u2s_var_hyd,
        "u1s_var_pow": u1s_var_pow,
        "u2s_var_pow": u2s_var_pow,
        "u1s_gem": u1s_gem,
        "u2s_gem": u2s_gem,
        "beta_1s": beta_1s_opt,
        "beta_2s": beta_2s_opt,
        "beta_1s_hyd": beta_1s_hyd_opt,
        "beta_2s_hyd": beta_2s_hyd_opt,
        "errors_1s": [
            abs(e1s_pert_1st - e1s_fd),
            abs(e1s_pert_2nd - e1s_fd),
            abs(e1s_var - e1s_fd),
            abs(e1s_var_hyd - e1s_fd),
            abs(e1s_var_pow - e1s_fd),
            abs(e1s_gem - e1s_fd),
            abs(e1s_shoot - e1s_fd),
        ],
        "errors_2s": [
            abs(e2s_pert_1st - e2s_fd),
            abs(e2s_pert_2nd - e2s_fd),
            abs(e2s_var - e2s_fd),
            abs(e2s_var_hyd - e2s_fd),
            abs(e2s_var_pow - e2s_fd),
            abs(e2s_gem - e2s_fd),
            abs(e2s_shoot - e2s_fd),
        ],
        "methods": [
            "Perturbation (1st)",
            "Perturbation (2nd)",
            "Variational (Harmonic)",
            "Variational (Hydrogenic)",
            "Variational (Power)",
            "Variational (GEM)",
            "Shooting IVP",
        ],
        "energies_1s": energies_1s,
        "energies_2s": energies_2s,
        "masses_1s": masses_1s,
        "masses_2s": masses_2s,
        "mass_diffs_1s": mass_diffs_1s,
        "mass_diffs_2s": mass_diffs_2s,
        "mass_pct_errors_1s": mass_pct_errors_1s,
        "mass_pct_errors_2s": mass_pct_errors_2s,
        "methods_all": methods_all,
    }


if __name__ == "__main__":
    run_comparisons()
