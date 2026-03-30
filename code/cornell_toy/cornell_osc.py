import math
import warnings
import numpy as np
from scipy.optimize import minimize, root_scalar, differential_evolution
from scipy.integrate import solve_ivp, simpson
from scipy.linalg import eigh_tridiagonal

# --- System Parameters ---
HBAR = 1.0
M_Q = 4.7  # Constituent quark mass (e.g., 4.7 GeV for bottom quark)
MU = M_Q / 2.0  # Reduced mass of the two-body quark-antiquark system
ALPHA = 0.4  # Coulomb-like interaction strength
B = 0.183  # Linear confinement strength
C = 0.0  # Constant potential shift (tune to ~0.11 to exactly match 9.4 GeV)
EPSILON = 0.1  # Softening parameter to avoid singularity at origin


def v_cornell(x_val):
    return -(4.0 / 3.0) * ALPHA / np.sqrt(x_val**2 + EPSILON**2) + B * np.abs(x_val) + C


def run_comparisons():
    print(
        f"1D Softened Cornell Potential: V(x) = -(4/3)*{ALPHA}/sqrt(x^2 + {EPSILON}^2) + {B}*|x| + {C}"
    )
    print(f"Parameters: m_q={M_Q} (Reduced mass mu={MU}), hbar={HBAR}\n")

    # ==========================================
    # GRID SETUP
    # ==========================================
    L = 15.0 / max(0.5, B)
    N = 6000
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    v_total = v_cornell(x)

    # ==========================================
    # 1. PERTURBATION THEORY (NUMERICAL)
    # ==========================================
    # We perturb around the softened Coulomb potential with the linear term.
    # The unperturbed eigenstates are found numerically using the finite difference method.
    v_unperturbed = -(4.0 / 3.0) * ALPHA / np.sqrt(x**2 + EPSILON**2) + C
    v_pert = B * np.abs(x)

    # Solve for unperturbed system
    diag_kin = HBAR**2 / (MU * dx**2)
    off_diag_kin = -(HBAR**2) / (2 * MU * dx**2)
    main_diag_unperturbed = diag_kin + v_unperturbed
    off_diag = np.full(N - 1, off_diag_kin)
    evals_unperturbed, evecs_unperturbed = eigh_tridiagonal(
        main_diag_unperturbed, off_diag
    )

    # Normalize eigenvectors to get wavefunctions
    psi_unperturbed = evecs_unperturbed / np.sqrt(dx)

    def calc_numerical_pert_energy(n, max_states=15):
        e_0 = evals_unperturbed[n]
        psi_n = psi_unperturbed[:, n]

        # 1st order correction: E_1 = <n|V'|n>
        integrand_1st = psi_n**2 * v_pert
        e_1 = simpson(y=integrand_1st, x=x)

        # 2nd order correction: E_2 = sum_{k!=n} |<k|V'|n>|^2 / (E_n - E_k)
        e_2 = 0.0
        for k in range(max_states):
            if k == n:
                continue

            psi_k = psi_unperturbed[:, k]
            e_k = evals_unperturbed[k]

            # H_kn = <k|V'|n>
            integrand_2nd = psi_k * v_pert * psi_n
            h_kn = simpson(y=integrand_2nd, x=x)

            if abs(h_kn) > 1e-10:
                e_2 += h_kn**2 / (e_0 - e_k)

        return e_0 + e_1, e_0 + e_1 + e_2

    e0_pert_1st, e0_pert_2nd = calc_numerical_pert_energy(0)
    e1_pert_1st, e1_pert_2nd = calc_numerical_pert_energy(1)

    # ==========================================
    # 2. VARIATIONAL METHOD (GENERALIZED)
    # ==========================================
    def compute_expectation_value(ansatz, params, x_grid, v_grid):
        psi = ansatz(x_grid, *params)
        norm = simpson(y=psi**2, x=x_grid)
        if norm <= 0:
            return np.inf
        pe = simpson(y=psi**2 * v_grid, x=x_grid) / norm
        dpsi_dx = np.gradient(psi, x_grid)
        ke = (HBAR**2 / (2 * MU)) * simpson(y=dpsi_dx**2, x=x_grid) / norm
        return ke + pe

    def ansatz_0_gauss(x_val, alpha_param):
        return np.exp(-alpha_param * x_val**2)

    def ansatz_1_gauss(x_val, alpha_param):
        return x_val * np.exp(-alpha_param * x_val**2)

    def ansatz_0_exp(x_val, alpha_param):
        return np.exp(-alpha_param * np.abs(x_val))

    def ansatz_1_exp(x_val, alpha_param):
        return x_val * np.exp(-alpha_param * np.abs(x_val))

    def ansatz_0_poly(x_val, alpha_param, *coeffs):
        poly = np.ones_like(x_val)
        for i, c in enumerate(coeffs):
            poly = poly + c * x_val ** (2 * (i + 1))
        return np.exp(-alpha_param * np.abs(x_val)) * poly

    def ansatz_1_poly(x_val, alpha_param, *coeffs):
        poly = np.copy(x_val)
        for i, c in enumerate(coeffs):
            poly = poly + c * x_val ** (2 * (i + 1) + 1)
        return np.exp(-alpha_param * np.abs(x_val)) * poly

    def ansatz_0_poly_gauss(x_val, alpha_param, *coeffs):
        poly = np.ones_like(x_val)
        for i, c in enumerate(coeffs):
            poly = poly + c * x_val ** (2 * (i + 1))
        return np.exp(-alpha_param * x_val**2) * poly

    def ansatz_1_poly_gauss(x_val, alpha_param, *coeffs):
        poly = np.copy(x_val)
        for i, c in enumerate(coeffs):
            poly = poly + c * x_val ** (2 * (i + 1) + 1)
        return np.exp(-alpha_param * x_val**2) * poly

    res0_gauss = minimize(
        lambda p: compute_expectation_value(ansatz_0_gauss, p, x, v_total),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    alpha0_gauss_opt = res0_gauss.x[0]
    e0_var_gauss = res0_gauss.fun

    res1_gauss = minimize(
        lambda p: compute_expectation_value(ansatz_1_gauss, p, x, v_total),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    alpha1_gauss_opt = res1_gauss.x[0]
    e1_var_gauss = res1_gauss.fun

    res0_exp = minimize(
        lambda p: compute_expectation_value(ansatz_0_exp, p, x, v_total),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    alpha0_exp_opt = res0_exp.x[0]
    e0_var_exp = res0_exp.fun

    res1_exp = minimize(
        lambda p: compute_expectation_value(ansatz_1_exp, p, x, v_total),
        x0=[1.0],
        bounds=[(0.01, 20.0)],
    )
    alpha1_exp_opt = res1_exp.x[0]
    e1_var_exp = res1_exp.fun

    # Global bounds for differential evolution: (alpha, c1, c2, c3, c4)
    poly_bounds = [
        (0.01, 20.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
    ]

    res0_poly = differential_evolution(
        lambda p: compute_expectation_value(ansatz_0_poly, p, x, v_total),
        bounds=poly_bounds,
    )
    params0_poly = res0_poly.x
    e0_var_poly = res0_poly.fun

    res1_poly = differential_evolution(
        lambda p: compute_expectation_value(ansatz_1_poly, p, x, v_total),
        bounds=poly_bounds,
    )
    params1_poly = res1_poly.x
    e1_var_poly = res1_poly.fun

    res0_poly_gauss = differential_evolution(
        lambda p: compute_expectation_value(ansatz_0_poly_gauss, p, x, v_total),
        bounds=poly_bounds,
    )
    params0_poly_gauss = res0_poly_gauss.x
    e0_var_poly_gauss = res0_poly_gauss.fun

    res1_poly_gauss = differential_evolution(
        lambda p: compute_expectation_value(ansatz_1_poly_gauss, p, x, v_total),
        bounds=poly_bounds,
    )
    params1_poly_gauss = res1_poly_gauss.x
    e1_var_poly_gauss = res1_poly_gauss.fun

    # ==========================================
    # 3. NUMERICAL METHOD 1: FINITE DIFFERENCE
    # ==========================================
    diag_kin = HBAR**2 / (MU * dx**2)
    off_diag_kin = -(HBAR**2) / (2 * MU * dx**2)

    main_diag = diag_kin + v_total
    off_diag = np.full(N - 1, off_diag_kin)

    evals_fd, evecs_fd = eigh_tridiagonal(main_diag, off_diag)
    e0_fd = evals_fd[0]
    e1_fd = evals_fd[1]

    # ==========================================
    # 4. NUMERICAL METHOD 2: SHOOTING METHOD
    # ==========================================
    def schrodinger_ode(x_val, y, E):
        psi, dpsi = y
        V = v_cornell(x_val)
        d2psi = (2 * MU / HBAR**2) * (V - E) * psi
        return [dpsi, d2psi]

    def shoot(E, parity):
        if parity == "even":
            y0 = [1.0, 0.0]
        else:
            y0 = [0.0, 1.0]

        # Terminate integration if psi diverges to infinity to avoid float overflow
        def divergence_event(x_val, y, E):
            return 1e50 - np.abs(y[0])

        divergence_event.terminal = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_ivp(
                schrodinger_ode,
                [0, L],
                y0,
                args=(E,),
                rtol=1e-8,
                atol=1e-11,
                events=divergence_event,
            )
        return sol.y[0, -1]

    # Dynamically scale the bracket margin to prevent bracketing failures
    bracket_margin0 = max(0.5, abs(e0_fd) * 0.05)
    root0 = root_scalar(
        shoot,
        args=("even",),
        bracket=[e0_fd - bracket_margin0, e0_fd + bracket_margin0],
    )
    e0_shoot = root0.root

    bracket_margin1 = max(0.5, abs(e1_fd) * 0.05)
    root1 = root_scalar(
        shoot, args=("odd",), bracket=[e1_fd - bracket_margin1, e1_fd + bracket_margin1]
    )
    e1_shoot = root1.root

    # ==========================================
    # 5. ERROR ANALYSIS
    # ==========================================
    print("--- Energy Eigenvalue Comparison Table ---")
    header = f"{'Calculation Method':<25} | {'E_0 (Ground State)':<20} | {'E_1 (1st Excited)'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(
        f"{'Perturbation Theory (1st)':<25} | {e0_pert_1st:<20.6f} | {e1_pert_1st:.6f}"
    )
    print(
        f"{'Perturbation Theory (2nd)':<25} | {e0_pert_2nd:<20.6f} | {e1_pert_2nd:.6f}"
    )
    print(
        f"{'Variational (Gaussian)':<25} | {e0_var_gauss:<20.6f} | {e1_var_gauss:.6f}"
    )
    print(f"{'Variational (Exponential)':<25} | {e0_var_exp:<20.6f} | {e1_var_exp:.6f}")
    print(f"{'Variational (Poly-Exp)':<25} | {e0_var_poly:<20.6f} | {e1_var_poly:.6f}")
    print(
        f"{'Variational (Poly-Gauss)':<25} | {e0_var_poly_gauss:<20.6f} | {e1_var_poly_gauss:.6f}"
    )
    print(f"{'Numerical: Matrix FD':<25} | {e0_fd:<20.6f} | {e1_fd:.6f}")
    print(f"{'Numerical: Shooting IVP':<25} | {e0_shoot:<20.6f} | {e1_shoot:.6f}")

    print("\n--- Physical Meson Mass Estimates (M = 2*m_q + E) ---")
    print(
        "Note: In 1D, the *odd* parity state (E_1) mathematically corresponds to the physical 3D S-wave ground state (u(0)=0)."
    )
    print(
        f"1D 1st Excited (3D 1S Ground State - e.g. eta_b): {2 * M_Q + e1_var_poly_gauss:.4f} GeV"
    )
    print(
        f"1D Ground State (Unphysical in 3D):               {2 * M_Q + e0_var_poly_gauss:.4f} GeV\n"
    )

    return {
        "x": x,
        "dx": dx,
        "v_total": v_total,
        "e0_fd": e0_fd,
        "e1_fd": e1_fd,
        "psi0_fd": evecs_fd[:, 0] / np.sqrt(dx),
        "psi1_fd": evecs_fd[:, 1] / np.sqrt(dx),
        "alpha0": alpha0_gauss_opt,
        "alpha1": alpha1_gauss_opt,
        "alpha0_exp": alpha0_exp_opt,
        "alpha1_exp": alpha1_exp_opt,
        "params0_poly": params0_poly,
        "params1_poly": params1_poly,
        "params0_poly_gauss": params0_poly_gauss,
        "params1_poly_gauss": params1_poly_gauss,
        "errors_e0": [
            abs(e0_pert_1st - e0_fd),
            abs(e0_pert_2nd - e0_fd),
            abs(e0_var_gauss - e0_fd),
            abs(e0_var_exp - e0_fd),
            abs(e0_var_poly - e0_fd),
            abs(e0_var_poly_gauss - e0_fd),
            abs(e0_shoot - e0_fd),
        ],
        "errors_e1": [
            abs(e1_pert_1st - e1_fd),
            abs(e1_pert_2nd - e1_fd),
            abs(e1_var_gauss - e1_fd),
            abs(e1_var_exp - e1_fd),
            abs(e1_var_poly - e1_fd),
            abs(e1_var_poly_gauss - e1_fd),
            abs(e1_shoot - e1_fd),
        ],
        "methods": [
            "Perturbation (1st)",
            "Perturbation (2nd)",
            "Variational (Gauss)",
            "Variational (Exp)",
            "Variational (Poly-Exp)",
            "Variational (Poly-Gauss)",
            "Shooting IVP",
        ],
        "energies_e0": [
            e0_pert_1st,
            e0_pert_2nd,
            e0_var_gauss,
            e0_var_exp,
            e0_var_poly,
            e0_var_poly_gauss,
            e0_shoot,
            e0_fd,
        ],
        "energies_e1": [
            e1_pert_1st,
            e1_pert_2nd,
            e1_var_gauss,
            e1_var_exp,
            e1_var_poly,
            e1_var_poly_gauss,
            e1_shoot,
            e1_fd,
        ],
        "methods_all": [
            "Perturbation (1st)",
            "Perturbation (2nd)",
            "Variational (Gauss)",
            "Variational (Exp)",
            "Variational (Poly-Exp)",
            "Variational (Poly-Gauss)",
            "Shooting IVP",
            "Numerical: Matrix FD",
        ],
    }


if __name__ == "__main__":
    run_comparisons()
