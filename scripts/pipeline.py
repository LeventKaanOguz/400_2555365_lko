import numpy as np
from scipy.optimize import minimize
from scripts.variational import normalize_ansatz, compute_expectation_value, solve_gem


def v_cornell_3d(r_val, alpha_s, b, c):
    return -(4.0 / 3.0) * alpha_s / r_val + b * r_val + c


def v_cornell_3d_smeared(r_val, m_q, alpha_s, b, c, sigma_smear, spin=0):
    v_base = v_cornell_3d(r_val, alpha_s, b, c)
    spin_dot = -0.75 if spin == 0 else 0.25
    hf_coeff = (32.0 * np.pi * alpha_s) / (9.0 * m_q**2)
    smeared_delta = (sigma_smear / np.sqrt(np.pi)) ** 3 * np.exp(
        -(sigma_smear**2) * r_val**2
    )
    return v_base + hf_coeff * spin_dot * smeared_delta


def run_cornell_pipeline(
    m_q,
    alpha_s,
    b,
    c,
    sigma_smear,
    exp_mass_1s,
    exp_mass_1s_triplet,
    exp_mass_2s,
    hbar=1.0,
    system_name="Quarkonium",
):
    mu = m_q / 2.0
    L = 15.0 / max(0.5, b)
    N = 6000
    r = np.linspace(L / N, L, N)
    dr = r[1] - r[0]

    print("=" * 85)
    print(f"3D CORNELL POTENTIAL ANALYSIS: {system_name.upper()}".center(85))
    print("=" * 85)
    print("\n--- System & Potential Parameters ---")
    print(f"Potential Model          : V(r) = -(4/3)*alpha_s/r + b*r + c")
    print(f"Quark Mass (m_q)         : {m_q:.3f} GeV")
    print(f"Reduced Mass (mu)        : {mu:.3f} GeV")
    print(f"Strong Coupling (alpha_s): {alpha_s:.3f}")
    print(f"String Tension (b)       : {b:.3f} GeV^2")
    print(f"Potential Shift (c)      : {c:.3f} GeV")
    print(f"Planck Constant (hbar)   : {hbar:.1f}")
    print(
        f"Grid Setup               : {N} points from r={r[0]:.4f} to r={r[-1]:.4f} (dr={dr:.4f})\n"
    )

    print("--- Experimental Reference Masses ---")
    print(f"1S State (S=0)           : {exp_mass_1s:.4f} GeV")
    print(f"1S State (S=1)           : {exp_mass_1s_triplet:.4f} GeV")
    print(f"2S State (S=1)           : {exp_mass_2s:.4f} GeV\n")

    print("--- Methodology Details ---")
    print(
        "1. Variational Method             : Optimizes trial wavefunctions (Harmonic, Hydrogenic,"
    )
    print("   Power) non-perturbatively using a smeared Gaussian spin-spin potential.")
    print(
        "2. Gaussian Expansion Method (GEM): Solves the Generalized Eigenvalue Problem using 25"
    )
    print("   non-orthogonal Gaussian basis functions distributed geometrically.\n")

    print("Note: Spin-spin hyperfine splitting is applied perturbatively for GEM, but")
    print(
        "non-perturbatively (smeared) for the Variational Method. S=0 is used for the 1S singlet,"
    )
    print("and S=1 for the 1S and 2S triplet states.\n")

    v_total = v_cornell_3d(r, alpha_s, b, c)

    def calc_hf_shift(u_array, spin=0):
        from scipy.integrate import simpson

        spin_dot = -0.75 if spin == 0 else 0.25
        hf_coeff = (32.0 * np.pi * alpha_s) / (9.0 * m_q**2)
        smeared_delta = (sigma_smear / np.sqrt(np.pi)) ** 3 * np.exp(
            -(sigma_smear**2) * r**2
        )
        return simpson(y=(u_array**2) * hf_coeff * spin_dot * smeared_delta, x=r)

    def calc_so_shift(u_array, l, s, j):
        """
        Calculates the Spin-Orbit (LS) correction for L > 0 states.
        Uses the typical Cornell V_LS(r) = 1/(2 m_q^2) * (4 alpha_s / r^3 - b / r).
        """
        if l == 0 or s == 0:
            return 0.0

        from scipy.integrate import simpson

        ls_dot = 0.5 * (j * (j + 1) - l * (l + 1) - s * (s + 1))

        # Avoid division by zero if r starts at 0. (Our grid starts at dr, so it's safe).
        v_ls = (1.0 / (2.0 * m_q**2)) * ((4.0 * alpha_s) / r**3 - b / r)

        shift = simpson(y=(u_array**2) * v_ls, x=r)
        return shift * ls_dot

    # ==========================================
    # 1. VARIATIONAL METHOD (Radial Ansatz)
    # ==========================================
    v_smeared_spin0 = v_cornell_3d_smeared(r, m_q, alpha_s, b, c, sigma_smear, spin=0)
    v_smeared_spin1 = v_cornell_3d_smeared(r, m_q, alpha_s, b, c, sigma_smear, spin=1)

    def ansatz_1s(r_val, beta):
        return r_val * np.exp(-0.5 * beta**2 * r_val**2)

    def ansatz_2s(r_val, beta):
        return r_val * (r_val**2 - 1.5 / beta**2) * np.exp(-0.5 * beta**2 * r_val**2)

    def ansatz_1s_hyd(r_val, beta):
        return r_val * np.exp(-beta * r_val)

    def ansatz_2s_hyd(r_val, beta):
        return r_val * (2.0 - beta * r_val) * np.exp(-0.5 * beta * r_val)

    def ansatz_1s_pow(r_val, r_n):
        return r_val * (r_val**0) * np.exp(-((r_val / r_n) ** 2))

    def ansatz_2s_pow(r_val, r_n):
        return r_val * (r_val**1) * np.exp(-((r_val / r_n) ** 2))

    res1s = minimize(
        lambda p: compute_expectation_value(ansatz_1s, p, r, v_smeared_spin0, hbar, mu),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_1s_opt = res1s.x[0]
    e1s_var = res1s.fun
    u1s_var = normalize_ansatz(ansatz_1s, [beta_1s_opt], r)

    res2s = minimize(
        lambda p: compute_expectation_value(ansatz_2s, p, r, v_smeared_spin1, hbar, mu),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_2s_opt = res2s.x[0]
    e2s_var = res2s.fun
    u2s_var = normalize_ansatz(ansatz_2s, [beta_2s_opt], r)

    res1s_hyd = minimize(
        lambda p: compute_expectation_value(
            ansatz_1s_hyd, p, r, v_smeared_spin0, hbar, mu
        ),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_1s_hyd_opt = res1s_hyd.x[0]
    e1s_var_hyd = res1s_hyd.fun
    u1s_var_hyd = normalize_ansatz(ansatz_1s_hyd, [beta_1s_hyd_opt], r)

    res2s_hyd = minimize(
        lambda p: compute_expectation_value(
            ansatz_2s_hyd, p, r, v_smeared_spin1, hbar, mu
        ),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_2s_hyd_opt = res2s_hyd.x[0]
    e2s_var_hyd = res2s_hyd.fun
    u2s_var_hyd = normalize_ansatz(ansatz_2s_hyd, [beta_2s_hyd_opt], r)

    res1s_pow = minimize(
        lambda p: compute_expectation_value(
            ansatz_1s_pow, p, r, v_smeared_spin0, hbar, mu
        ),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    rn_1s_opt = res1s_pow.x[0]
    e1s_var_pow = res1s_pow.fun
    u1s_var_pow = normalize_ansatz(ansatz_1s_pow, [rn_1s_opt], r)

    res2s_pow = minimize(
        lambda p: compute_expectation_value(
            ansatz_2s_pow, p, r, v_smeared_spin1, hbar, mu
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
    # 2. GAUSSIAN EXPANSION METHOD (GEM)
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
        hbar,
        mu,
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
    # SUMMARY TABLE
    # ==========================================
    print("--- Energy Eigenvalue Comparison Table ---")
    header = f"{'Calculation Method':<28} | {'E_1S (Ground State)':<20} | {'E_2S (1st Excited)'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'Variational (Harmonic)':<28} | {e1s_var:<20.6f} | {e2s_var:.6f}")
    print(
        f"{'Variational (Hydrogenic)':<28} | {e1s_var_hyd:<20.6f} | {e2s_var_hyd:.6f}"
    )
    print(f"{'Variational (Power)':<28} | {e1s_var_pow:<20.6f} | {e2s_var_pow:.6f}")
    print(f"{'Variational (GEM)':<28} | {e1s_gem:<20.6f} | {e2s_gem:.6f}")

    print("\n--- Physical Meson Mass Estimates (M = 2*m_q + E) ---")
    mass_header = f"{'Calculation Method':<28} | {'Mass 1S [GeV]':<15} | {'Diff [MeV]':<12} | {'% Error':<10} | {'Mass 2S [GeV]':<15} | {'Diff [MeV]':<12} | {'% Error':<10}"
    print("-" * len(mass_header))
    print(mass_header)
    print("-" * len(mass_header))

    methods_all = [
        "Variational (Harmonic)",
        "Variational (Hydrogenic)",
        "Variational (Power)",
        "Variational (GEM)",
    ]
    energies_1s = [
        e1s_var,
        e1s_var_hyd,
        e1s_var_pow,
        e1s_gem,
    ]
    energies_2s = [
        e2s_var,
        e2s_var_hyd,
        e2s_var_pow,
        e2s_gem,
    ]

    masses_1s = [2 * m_q + e for e in energies_1s]
    masses_2s = [2 * m_q + e for e in energies_2s]
    mass_diffs_1s = [(m - exp_mass_1s) * 1000 for m in masses_1s]
    mass_diffs_2s = [(m - exp_mass_2s) * 1000 for m in masses_2s]
    mass_pct_errors_1s = [abs(m - exp_mass_1s) / exp_mass_1s * 100 for m in masses_1s]
    mass_pct_errors_2s = [abs(m - exp_mass_2s) / exp_mass_2s * 100 for m in masses_2s]

    for i, method in enumerate(methods_all):
        m1, d1, p1 = masses_1s[i], mass_diffs_1s[i], mass_pct_errors_1s[i]
        m2, d2, p2 = masses_2s[i], mass_diffs_2s[i], mass_pct_errors_2s[i]
        print(
            f"{method:<28} | {m1:<15.4f} | {d1:<12.1f} | {p1:<10.3f} | {m2:<15.4f} | {d2:<12.1f} | {p2:<10.3f}"
        )
    print()

    # Orbital Excitations for general outputs
    v_eff_1 = v_total + 1 * 2 * hbar**2 / (2.0 * mu * r**2)
    v_eff_2 = v_total + 2 * 3 * hbar**2 / (2.0 * mu * r**2)

    evals_gem_1, u_gem_1, evecs_gem_1, nu_gem_1 = solve_gem(
        r, v_eff_1, hbar, mu, n_basis=n_basis_gem, r_min=r_min_gem, r_max=r_max_gem, l=1
    )
    evals_gem_2, u_gem_2, evecs_gem_2, nu_gem_2 = solve_gem(
        r, v_eff_2, hbar, mu, n_basis=n_basis_gem, r_min=r_min_gem, r_max=r_max_gem, l=2
    )

    return {
        "r": r,
        "dr": dr,
        "v_total": v_total,
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
        "methods": methods_all,
        "energies_1s": energies_1s,
        "energies_2s": energies_2s,
        "masses_1s": masses_1s,
        "masses_2s": masses_2s,
        "mass_diffs_1s": mass_diffs_1s,
        "mass_diffs_2s": mass_diffs_2s,
        "mass_pct_errors_1s": mass_pct_errors_1s,
        "mass_pct_errors_2s": mass_pct_errors_2s,
        "methods_all": methods_all,
        "evals_gem": evals_gem,
        "u_gem": u_gem,
        "evals_gem_1": evals_gem_1,
        "u_gem_1": u_gem_1,
        "evecs_gem_1": evecs_gem_1,
        "nu_gem_1": nu_gem_1,
        "evals_gem_2": evals_gem_2,
        "u_gem_2": u_gem_2,
        "evecs_gem_2": evecs_gem_2,
        "nu_gem_2": nu_gem_2,
        "evecs_gem": evecs_gem,
        "nu_gem": nu_gem,
        "n_basis_gem": n_basis_gem,
        "calc_hf_shift": calc_hf_shift,
        "calc_so_shift": calc_so_shift,
    }
