import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eigh


def normalize_ansatz(ansatz, params, x_grid):
    """
    Normalize an unnormalized variational ansatz wavefunction.

    Parameters
    ----------
    ansatz : callable
        The unnormalized trial wavefunction function.
    params : list
        Parameters to pass to the ansatz function.
    x_grid : numpy.ndarray
        The spatial grid array.

    Returns
    -------
    numpy.ndarray or None
        The normalized wavefunction array, or None if not normalizable (norm <= 0).
    """
    psi = ansatz(x_grid, *params)
    norm = np.sqrt(simpson(y=psi**2, x=x_grid))
    if norm <= 0:
        return None
    return psi / norm


def compute_expectation_value(ansatz, params, x_grid, v_grid, hbar, mass):
    """
    Compute the expected total energy value for a given variational ansatz.

    Calculates the expectation values of kinetic and potential energies
    using numerical integration over the spatial grid.

    Parameters
    ----------
    ansatz : callable
        The unnormalized trial wavefunction function.
    params : list
        Parameters to evaluate the ansatz function.
    x_grid : numpy.ndarray
        The spatial grid array.
    v_grid : numpy.ndarray
        The potential energy evaluated on the spatial grid.
    hbar : float
        Reduced Planck constant.
    mass : float
        Mass of the particle (or reduced mass of the system).

    Returns
    -------
    float
        The total expected energy value, or infinity if not normalizable.
    """
    psi = ansatz(x_grid, *params)
    norm = simpson(y=psi**2, x=x_grid)
    if norm <= 0:
        return np.inf
    pe = simpson(y=psi**2 * v_grid, x=x_grid) / norm
    dpsi_dx = np.gradient(psi, x_grid)
    ke = (hbar**2 / (2 * mass)) * simpson(y=dpsi_dx**2, x=x_grid) / norm
    return ke + pe


def ansatz_0_gauss(x_val, alpha):
    return np.exp(-alpha * x_val**2)


def ansatz_1_gauss(x_val, alpha):
    return x_val * np.exp(-alpha * x_val**2)


def ansatz_0_exp(x_val, alpha):
    return np.exp(-alpha * np.abs(x_val))


def ansatz_1_exp(x_val, alpha):
    return x_val * np.exp(-alpha * np.abs(x_val))


def build_even_poly(x_val, coeffs):
    poly = np.ones_like(x_val)
    for i, c in enumerate(coeffs):
        poly = poly + c * x_val ** (2 * (i + 1))
    return poly


def build_odd_poly(x_val, coeffs):
    poly = np.copy(x_val)
    for i, c in enumerate(coeffs):
        poly = poly + c * x_val ** (2 * (i + 1) + 1)
    return poly


def ansatz_0_poly(x_val, alpha, *coeffs):
    return np.exp(-alpha * np.abs(x_val)) * build_even_poly(x_val, coeffs)


def ansatz_1_poly(x_val, alpha, *coeffs):
    return np.exp(-alpha * np.abs(x_val)) * build_odd_poly(x_val, coeffs)


def ansatz_0_poly_gauss(x_val, alpha, *coeffs):
    return np.exp(-alpha * x_val**2) * build_even_poly(x_val, coeffs)


def ansatz_1_poly_gauss(x_val, alpha, *coeffs):
    return np.exp(-alpha * x_val**2) * build_odd_poly(x_val, coeffs)


def optimize_ansatz(
    ansatz, x_grid, v_grid, hbar, mass, x0=None, bounds=None, method="minimize"
):
    """
    Optimize the parameters of a variational ansatz to minimize expected energy.

    Parameters
    ----------
    ansatz : callable
        The trial wavefunction function.
    x_grid : numpy.ndarray
        The spatial grid array.
    v_grid : numpy.ndarray
        The potential energy array.
    hbar : float
        Reduced Planck constant.
    mass : float
        Mass of the particle.
    x0 : list, optional
        Initial parameter guess for the optimizer.
    bounds : list of tuples, optional
        Bounds for the parameters to constrain optimization.
    method : str, optional
        Optimization routine to use ('minimize' or 'differential_evolution').

    Returns
    -------
    tuple
        (optimized_params, minimized_energy): The best parameters and resulting energy.
    """
    if method == "minimize":
        res = minimize(
            lambda p: compute_expectation_value(ansatz, p, x_grid, v_grid, hbar, mass),
            x0=x0,
            bounds=bounds,
        )
    elif method == "differential_evolution":
        res = differential_evolution(
            lambda p: compute_expectation_value(ansatz, p, x_grid, v_grid, hbar, mass),
            bounds=bounds,
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    return res.x, res.fun


def solve_gem(x_grid, v_grid, hbar, mass, n_basis, r_min, r_max, l=0):
    """
    Solves the radial Schrödinger equation using the Gaussian Expansion Method (GEM).

    Basis parameters `nu_i` are distributed geometrically between `r_min` and `r_max`
    to accurately model both short-range interactions and long-range tails.

    Parameters
    ----------
    x_grid : numpy.ndarray
        The spatial grid array.
    v_grid : numpy.ndarray
        The potential energy array evaluated on the grid.
    hbar : float
        Reduced Planck constant.
    mass : float
        Mass of the particle (or reduced mass).
    n_basis : int
        The number of Gaussian basis functions to construct.
    r_min : float
        The minimum radial limit parameter for basis geometric scaling.
    r_max : float
        The maximum radial limit parameter for basis geometric scaling.
    l : int, optional
        Orbital angular momentum quantum number. Default is 0.

    Returns
    -------
    tuple
        (evals, wavefunctions): Arrays of eigenvalues and corresponding eigenvectors (columns).
    """
    # 1. Setup the Gaussian widths (geometric progression)
    a = (r_max / r_min) ** (2.0 / (n_basis - 1)) if n_basis > 1 else 1.0
    nu = [1.0 / (r_min**2 * a**i) for i in range(n_basis)]

    # 2. Construct non-orthogonal basis functions (reduced radial u_i(r) = r * R_i(r))
    # For general angular momentum l, R(r) ~ r^l e^{-nu r^2}, so u(r) ~ r^{l+1} e^{-nu r^2}
    basis = []
    dbasis = []
    for n in nu:
        u = x_grid ** (l + 1) * np.exp(-n * x_grid**2)
        du = ((l + 1) * x_grid**l - 2 * n * x_grid ** (l + 2)) * np.exp(-n * x_grid**2)

        # Normalize individual basis elements numerically for stability
        norm = np.sqrt(simpson(y=u**2, x=x_grid))
        basis.append(u / norm)
        dbasis.append(du / norm)

    # 3. Compute Hamiltonian (H) and Overlap (S) Matrices
    S = np.zeros((n_basis, n_basis))
    H = np.zeros((n_basis, n_basis))

    for i in range(n_basis):
        for j in range(i, n_basis):
            # Overlap S_ij
            S_ij = simpson(y=basis[i] * basis[j], x=x_grid)
            S[i, j] = S[j, i] = S_ij

            # Kinetic Energy T_ij
            T_ij = (hbar**2 / (2 * mass)) * simpson(y=dbasis[i] * dbasis[j], x=x_grid)

            # Potential Energy V_ij
            V_ij = simpson(y=basis[i] * v_grid * basis[j], x=x_grid)

            H_ij = T_ij + V_ij
            H[i, j] = H[j, i] = H_ij

    # 4. Solve the Generalized Eigenvalue Problem (H c = E S c)
    evals, evecs = eigh(H, S)

    # 5. Construct full eigenfunctions
    wavefunctions = np.zeros((len(x_grid), len(evals)))
    for state_idx in range(len(evals)):
        c = evecs[:, state_idx]
        u_full = sum(c[i] * basis[i] for i in range(n_basis))
        wavefunctions[:, state_idx] = normalize_ansatz(lambda r, *p: u_full, [], x_grid)

    return evals, wavefunctions
