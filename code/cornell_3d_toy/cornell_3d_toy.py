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
M_Q = 4.730  # Constituent quark mass (e.g., bottom quark)
MU = M_Q / 2.0  # Reduced mass of the two-body quark-antiquark system
ALPHA_S = 0.22  # Strong coupling constant for the Coulomb-like interaction
B = 0.18  # Linear confinement string tension
C = -0.07  # Constant potential shift
EXP_MASS_1S = 9.3987  # eta_b (1S) mass in GeV
EXP_MASS_2S = 10.0234  # Upsilon (2S) mass in GeV


def v_cornell_3d(r_val):
    return -(4.0 / 3.0) * ALPHA_S / r_val + B * r_val + C


def run_comparisons():
    print(f"3D Radial Cornell Potential: V(r) = -(4/3)*{ALPHA_S}/r + {B}*r + {C}")
    print(f"Parameters: m_q={M_Q} (Reduced mass mu={MU}), hbar={HBAR}\n")
    print(
        f"Experimental masses: eta_b(1S) = {EXP_MASS_1S} GeV, Upsilon(2S) = {EXP_MASS_2S} GeV"
    )
    print(
        "Note: Spin-spin hyperfine perturbation applied: S=0 (singlet) for 1S states, S=1 (triplet) for 2S states.\n"
    )

    # ==========================================
    # GRID SETUP
    # ==========================================
    # We avoid r=0 strictly. The boundary condition u(0)=0 is mathematically
    # accommodated by starting our grid at dr.
    L = 15.0 / max(0.5, B)
    N = 6000
    r = np.linspace(L / N, L, N)
    dr = r[1] - r[0]

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
        return r_val * (r_val**0) * np.exp(-(r_val / r_n)**2)

    def ansatz_2s_pow(r_val, r_n):
        return r_val * (r_val**1) * np.exp(-(r_val / r_n)**2)

    res1s = minimize(
        lambda p: compute_expectation_value(ansatz_1s, p, r, v_total, HBAR, MU),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_1s_opt = res1s.x[0]
    e1s_var = res1s.fun
    u1s_var = normalize_ansatz(ansatz_1s, [beta_1s_opt], r)
    e1s_var += calc_hf_shift(u1s_var, spin=0)

    res2s = minimize(
        lambda p: compute_expectation_value(ansatz_2s, p, r, v_total, HBAR, MU),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_2s_opt = res2s.x[0]
    e2s_var = res2s.fun
    u2s_var = normalize_ansatz(ansatz_2s, [beta_2s_opt], r)
    e2s_var += calc_hf_shift(u2s_var, spin=1)

    res1s_hyd = minimize(
        lambda p: compute_expectation_value(ansatz_1s_hyd, p, r, v_total, HBAR, MU),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_1s_hyd_opt = res1s_hyd.x[0]
    e1s_var_hyd = res1s_hyd.fun
    u1s_var_hyd = normalize_ansatz(ansatz_1s_hyd, [beta_1s_hyd_opt], r)
    e1s_var_hyd += calc_hf_shift(u1s_var_hyd, spin=0)

    res2s_hyd = minimize(
        lambda p: compute_expectation_value(ansatz_2s_hyd, p, r, v_total, HBAR, MU),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    beta_2s_hyd_opt = res2s_hyd.x[0]
    e2s_var_hyd = res2s_hyd.fun
    u2s_var_hyd = normalize_ansatz(ansatz_2s_hyd, [beta_2s_hyd_opt], r)
    e2s_var_hyd += calc_hf_shift(u2s_var_hyd, spin=1)

    res1s_pow = minimize(
        lambda p: compute_expectation_value(ansatz_1s_pow, p, r, v_total, HBAR, MU),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    rn_1s_opt = res1s_pow.x[0]
    e1s_var_pow = res1s_pow.fun
    u1s_var_pow = normalize_ansatz(ansatz_1s_pow, [rn_1s_opt], r)
    e1s_var_pow += calc_hf_shift(u1s_var_pow, spin=0)

    res2s_pow = minimize(
        lambda p: compute_expectation_value(ansatz_2s_pow, p, r, v_total, HBAR, MU),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    rn_2s_opt = res2s_pow.x[0]
    e2s_var_pow = res2s_pow.fun
    u2s_var_pow = normalize_ansatz(ansatz_2s_pow, [rn_2s_opt], r)
    e2s_var_pow += calc_hf_shift(u2s_var_pow, spin=1)

    # ==========================================
    # 3. GAUSSIAN EXPANSION METHOD (GEM)
    # ==========================================
    evals_gem, u_gem = solve_gem(
        r, v_total, HBAR, MU, n_basis=15, r_min=0.1, r_max=L, l=0
    )
    e1s_gem_bare = evals_gem[0]
    e2s_gem_bare = evals_gem[1]
    u1s_gem = u_gem[:, 0]
    u2s_gem = u_gem[:, 1]

    e1s_gem = e1s_gem_bare + calc_hf_shift(u1s_gem, spin=0)
    e2s_gem = e2s_gem_bare + calc_hf_shift(u2s_gem, spin=1)

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
    header = f"{'Calculation Method':<25} | {'E_1S (Ground State)':<20} | {'E_2S (1st Excited)'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(
        f"{'Perturbation Theory (1st)':<25} | {e1s_pert_1st:<20.6f} | {e2s_pert_1st:.6f}"
    )
    print(
        f"{'Perturbation Theory (2nd)':<25} | {e1s_pert_2nd:<20.6f} | {e2s_pert_2nd:.6f}"
    )
    print(f"{'Variational (Harmonic)':<25} | {e1s_var:<20.6f} | {e2s_var:.6f}")
    print(
        f"{'Variational (Hydrogenic)':<25} | {e1s_var_hyd:<20.6f} | {e2s_var_hyd:.6f}"
    )
    print(f"{'Variational (Power)':<25} | {e1s_var_pow:<20.6f} | {e2s_var_pow:.6f}")
    print(f"{'Variational (GEM)':<25} | {e1s_gem:<20.6f} | {e2s_gem:.6f}")
    print(f"{'Numerical: Matrix FD':<25} | {e1s_fd:<20.6f} | {e2s_fd:.6f}")
    print(f"{'Numerical: Shooting IVP':<25} | {e1s_shoot:<20.6f} | {e2s_shoot:.6f}")

    print("\n--- Physical Meson Mass Estimates (M = 2*m_q + E) ---")
    mass_header = f"{'Calculation Method':<25} | {'Mass 1S (eta_b) [GeV]':<25} | {'Diff [MeV]':<15} | {'Mass 2S (Upsilon) [GeV]':<25} | {'Diff [MeV]':<15}"
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

    for i, method in enumerate(methods_all):
        m1, d1 = masses_1s[i], mass_diffs_1s[i]
        m2, d2 = masses_2s[i], mass_diffs_2s[i]
        print(f"{method:<25} | {m1:<25.4f} | {d1:<15.1f} | {m2:<25.4f} | {d2:<15.1f}")
    print()

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
        "methods_all": methods_all,
    }


if __name__ == "__main__":
    run_comparisons()
