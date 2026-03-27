import math
import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.integrate import solve_ivp, simpson, quad
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_hermite

# --- System Parameters ---
# Feel free to change m, omega, and sigma to test different scenarios
HBAR = 1.0
M = 1.0
OMEGA = 1.0
SIGMA = 100


def run_comparisons():
    print(f"Quantum Harmonic Oscillator perturbed by V' = {SIGMA} * x^4")
    print(f"Parameters: m={M}, omega={OMEGA}, hbar={HBAR}\n")

    # ==========================================
    # 1. PERTURBATION THEORY (GENERALIZED)
    # ==========================================
    # We numerically derive the corrections using integrals over unperturbed eigenstates.
    # E_n^(1) = <n| V' |n>
    # E_n^(2) = sum_{k != n} |<k| V' |n>|^2 / (E_n^(0) - E_k^(0))

    def unperturbed_state(n, x_val):
        alpha = np.sqrt(M * OMEGA / HBAR)
        xi = alpha * x_val
        norm = np.sqrt(alpha / (math.sqrt(np.pi) * 2**n * math.factorial(n)))
        return norm * np.exp(-0.5 * xi**2) * eval_hermite(n, xi)

    def calc_pert_energy(n, v_pert_func, max_states=10):
        e_0 = HBAR * OMEGA * (n + 0.5)

        # Dynamically scale integration limits based on the wavefunction's width (20 std deviations)
        limit = 20.0 * np.sqrt(HBAR / (M * OMEGA))

        # 1st order correction
        def integrand_1st(x_val):
            psi = unperturbed_state(n, x_val)
            return psi**2 * v_pert_func(x_val)

        e_1, _ = quad(integrand_1st, -limit, limit)

        # 2nd order correction
        e_2 = 0.0
        for k in range(max_states):
            if k == n:
                continue

            def integrand_2nd(x_val):
                return (
                    unperturbed_state(k, x_val)
                    * v_pert_func(x_val)
                    * unperturbed_state(n, x_val)
                )

            h_kn, _ = quad(integrand_2nd, -limit, limit)

            # Add to 2nd order sum if the matrix element is non-zero
            if abs(h_kn) > 1e-10:
                e_k = HBAR * OMEGA * (k + 0.5)
                e_2 += (h_kn**2) / (e_0 - e_k)

        return e_0 + e_1, e_0 + e_1 + e_2

    # Evaluate generalized perturbation using V' = sigma * x^4
    v_pert = lambda x_val: SIGMA * x_val**4
    e0_pert_1st, e0_pert_2nd = calc_pert_energy(0, v_pert)
    e1_pert_1st, e1_pert_2nd = calc_pert_energy(1, v_pert)

    # ==========================================
    # GRID SETUP
    # ==========================================
    # Dynamically scale the spatial domain to prevent float overflow for large SIGMA.
    # A factor of (1 + SIGMA)**0.25 scales it based on the x^4 dominance.
    L = 8.0 / (1.0 + SIGMA) ** 0.25
    N = 4000  # number of grid points
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    # Potential energy terms
    v_total = 0.5 * M * OMEGA**2 * x**2 + SIGMA * x**4

    # ==========================================
    # 2. VARIATIONAL METHOD (GENERALIZED)
    # ==========================================
    def compute_expectation_value(ansatz, params, x_grid, v_grid):
        psi = ansatz(x_grid, *params)

        # Compute norm: <psi|psi>
        norm = simpson(y=psi**2, x=x_grid)
        if norm <= 0:
            return np.inf

        # Compute potential energy: <psi|V|psi>
        pe = simpson(y=psi**2 * v_grid, x=x_grid) / norm

        # Compute kinetic energy: <psi|T|psi>
        # Using integration by parts: (hbar^2/2m) \int (dpsi/dx)^2 dx
        dpsi_dx = np.gradient(psi, x_grid)
        ke = (HBAR**2 / (2 * M)) * simpson(y=dpsi_dx**2, x=x_grid) / norm

        return ke + pe

    # Define arbitrary ansatz functions (easily extensible to more parameters)
    def ansatz_0(x_val, alpha):
        return np.exp(-alpha * x_val**2)

    def ansatz_1(x_val, alpha):
        return x_val * np.exp(-alpha * x_val**2)

    # Minimize parameters using scipy's general optimizer
    res0 = minimize(
        lambda p: compute_expectation_value(ansatz_0, p, x, v_total),
        x0=[1.0],
        bounds=[(0.01, 10.0)],
    )
    alpha0_opt = res0.x[0]
    e0_var = res0.fun

    res1 = minimize(
        lambda p: compute_expectation_value(ansatz_1, p, x, v_total),
        x0=[1.0],
        bounds=[(0.01, 10.0)],
    )
    alpha1_opt = res1.x[0]
    e1_var = res1.fun

    print("--- Variational Method Results ---")
    print(f"Optimized alpha_0 (Ground State):      {alpha0_opt:.6f}")
    print(f"Optimized alpha_1 (1st Excited State): {alpha1_opt:.6f}\n")

    # ==========================================
    # 3. NUMERICAL METHOD 1: FINITE DIFFERENCE
    # ==========================================
    # Approximates the derivatives on a grid to create a sparse Hamiltonian matrix

    # Kinetic energy terms
    diag_kin = HBAR**2 / (M * dx**2)
    off_diag_kin = -(HBAR**2) / (2 * M * dx**2)

    # Construct diagonals
    main_diag = diag_kin + v_total
    off_diag = np.full(N - 1, off_diag_kin)

    # Solve eigenvalue problem for tridiagonal matrix
    evals_fd, evecs_fd = eigh_tridiagonal(main_diag, off_diag)
    e0_fd = evals_fd[0]
    e1_fd = evals_fd[1]

    # ==========================================
    # 4. NUMERICAL METHOD 2: SHOOTING METHOD
    # ==========================================
    # Integrates the Schrödinger ODE and searches for eigenvalues E
    # where the wavefunction psi(x) -> 0 as x -> infinity.

    def schrodinger_ode(x_val, y, E):
        psi, dpsi = y
        V = 0.5 * M * OMEGA**2 * x_val**2 + SIGMA * x_val**4
        # d^2(psi)/dx^2 = 2m/hbar^2 * (V - E) * psi
        d2psi = (2 * M / HBAR**2) * (V - E) * psi
        return [dpsi, d2psi]

    def shoot(E, parity):
        # Initial conditions at x=0 depending on parity
        if parity == "even":
            y0 = [1.0, 0.0]  # Ground state is even
        else:
            y0 = [0.0, 1.0]  # First excited state is odd

        # Integrate from x=0 to x=L
        sol = solve_ivp(schrodinger_ode, [0, L], y0, args=(E,), rtol=1e-8, atol=1e-11)
        # Return the boundary value at x=L (we want this to be 0)
        return sol.y[0, -1]

    # We use the FD results to provide very safe brackets for the root-finder
    root0 = root_scalar(shoot, args=("even",), bracket=[e0_fd - 0.1, e0_fd + 0.1])
    e0_shoot = root0.root

    root1 = root_scalar(shoot, args=("odd",), bracket=[e1_fd - 0.1, e1_fd + 0.1])
    e1_shoot = root1.root

    # ==========================================
    # SUMMARY TABLE
    # ==========================================
    # Print out nicely aligned comparisons
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
    print(f"{'Variational Method':<25} | {e0_var:<20.6f} | {e1_var:.6f}")
    print(f"{'Numerical: Matrix FD':<25} | {e0_fd:<20.6f} | {e1_fd:.6f}")
    print(f"{'Numerical: Shooting IVP':<25} | {e0_shoot:<20.6f} | {e1_shoot:.6f}")

    # ==========================================
    # 5. ERROR ANALYSIS
    # ==========================================
    print("\n--- Absolute Errors (vs Numerical Matrix FD) ---")
    error_header = f"{'Calculation Method':<25} | {'E_0 Error':<20} | {'E_1 Error'}"
    print("-" * len(error_header))
    print(error_header)
    print("-" * len(error_header))
    print(
        f"{'Perturbation Theory (1st)':<25} | {abs(e0_pert_1st - e0_fd):<20.6e} | {abs(e1_pert_1st - e1_fd):.6e}"
    )
    print(
        f"{'Perturbation Theory (2nd)':<25} | {abs(e0_pert_2nd - e0_fd):<20.6e} | {abs(e1_pert_2nd - e1_fd):.6e}"
    )
    print(
        f"{'Variational Method':<25} | {abs(e0_var - e0_fd):<20.6e} | {abs(e1_var - e1_fd):.6e}"
    )
    print(
        f"{'Numerical: Shooting IVP':<25} | {abs(e0_shoot - e0_fd):<20.6e} | {abs(e1_shoot - e1_fd):.6e}"
    )

    # Return the aggregated results to be used by the plotting script
    return {
        "x": x,
        "dx": dx,
        "v_total": v_total,
        "e0_fd": e0_fd,
        "e1_fd": e1_fd,
        # Normalize discrete eigenvectors to continuous probability density wavefunctions
        "psi0_fd": evecs_fd[:, 0] / np.sqrt(dx),
        "psi1_fd": evecs_fd[:, 1] / np.sqrt(dx),
        "alpha0": alpha0_opt,
        "alpha1": alpha1_opt,
        "errors_e0": [
            abs(e0_pert_1st - e0_fd),
            abs(e0_pert_2nd - e0_fd),
            abs(e0_var - e0_fd),
            abs(e0_shoot - e0_fd),
        ],
        "errors_e1": [
            abs(e1_pert_1st - e1_fd),
            abs(e1_pert_2nd - e1_fd),
            abs(e1_var - e1_fd),
            abs(e1_shoot - e1_fd),
        ],
        "methods": [
            "Perturbation (1st)",
            "Perturbation (2nd)",
            "Variational",
            "Shooting IVP",
        ],
        "energies_e0": [
            e0_pert_1st,
            e0_pert_2nd,
            e0_var,
            e0_shoot,
            e0_fd,
        ],
        "energies_e1": [
            e1_pert_1st,
            e1_pert_2nd,
            e1_var,
            e1_shoot,
            e1_fd,
        ],
        "methods_all": [
            "Perturbation (1st)",
            "Perturbation (2nd)",
            "Variational",
            "Shooting IVP",
            "Numerical: Matrix FD",
        ],
    }


if __name__ == "__main__":
    run_comparisons()
