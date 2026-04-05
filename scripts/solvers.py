import numpy as np
import warnings
from scipy.integrate import simpson, solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import root_scalar


def solve_fd(x, v_total, mass, hbar=1.0):
    """
    Solve the 1D or radial Schrödinger equation using the Finite Difference method.

    Parameters
    ----------
    x : numpy.ndarray
        The spatial grid array.
    v_total : numpy.ndarray
        The potential energy evaluated on the grid.
    mass : float
        Mass of the particle.
    hbar : float, optional
        Reduced Planck constant. Default is 1.0.

    Returns
    -------
    tuple
        (evals, evecs): Energy eigenvalues and corresponding eigenvectors.
    """
    dx = x[1] - x[0]
    N = len(x)
    diag_kin = hbar**2 / (mass * dx**2)
    off_diag_kin = -(hbar**2) / (2 * mass * dx**2)

    main_diag = diag_kin + v_total
    off_diag = np.full(N - 1, off_diag_kin)

    evals, evecs = eigh_tridiagonal(main_diag, off_diag)
    return evals, evecs


def calc_pert_energy_numerical(n, evals_unp, evecs_unp, v_pert, x, max_states=15):
    """
    Calculate 1st and 2nd order perturbation energy corrections numerically.

    Parameters
    ----------
    n : int
        The quantum state number to correct.
    evals_unp : numpy.ndarray
        Unperturbed energy eigenvalues.
    evecs_unp : numpy.ndarray
        Unperturbed eigenfunctions.
    v_pert : numpy.ndarray
        Perturbing potential array.
    x : numpy.ndarray
        The spatial grid array.
    max_states : int, optional
        Maximum number of states to iterate for 2nd order correction. Default is 15.

    Returns
    -------
    tuple
        (e_1st, e_2nd): Energies evaluated at 1st and 2nd perturbation orders.
    """
    e_0 = evals_unp[n]
    dx = x[1] - x[0]
    psi_unp = evecs_unp / np.sqrt(dx)
    psi_n = psi_unp[:, n]

    integrand_1st = psi_n**2 * v_pert
    e_1 = simpson(y=integrand_1st, x=x)

    e_2 = 0.0
    for k in range(max_states):
        if k == n:
            continue
        psi_k = psi_unp[:, k]
        e_k = evals_unp[k]

        integrand_2nd = psi_k * v_pert * psi_n
        h_kn = simpson(y=integrand_2nd, x=x)

        if abs(h_kn) > 1e-10:
            e_2 += h_kn**2 / (e_0 - e_k)

    return e_0 + e_1, e_0 + e_1 + e_2


def normalize_ansatz(ansatz, params, r_grid):
    """
    Normalize an unnormalized generic wavefunction ansatz.

    Parameters
    ----------
    ansatz : callable
        The unnormalized trial wavefunction function.
    params : list
        Parameters to evaluate the ansatz function.
    r_grid : numpy.ndarray
        The spatial radial grid array.

    Returns
    -------
    numpy.ndarray or None
        Normalized wavefunction, or None if the norm is less than or equal to zero.
    """
    u = ansatz(r_grid, *params)
    norm = np.sqrt(simpson(y=u**2, x=r_grid))
    if norm <= 0:
        return None
    return u / norm


def compute_expectation_value(ansatz, params, r_grid, v_grid, mass, hbar=1.0):
    """
    Compute the energy expectation value for a given variational ansatz.

    Parameters
    ----------
    ansatz : callable
        The unnormalized trial wavefunction function.
    params : list
        Parameters to evaluate the ansatz function.
    r_grid : numpy.ndarray
        The spatial radial grid array.
    v_grid : numpy.ndarray
        The potential energy array evaluated on the grid.
    mass : float
        Mass of the particle.
    hbar : float, optional
        Reduced Planck constant. Default is 1.0.

    Returns
    -------
    float
        The total expected energy value, or infinity if not normalizable.
    """
    u_norm = normalize_ansatz(ansatz, params, r_grid)
    if u_norm is None:
        return np.inf
    pe = simpson(y=u_norm**2 * v_grid, x=r_grid)
    du_dr = np.gradient(u_norm, r_grid)
    ke = (hbar**2 / (2 * mass)) * simpson(y=du_dr**2, x=r_grid)
    return ke + pe


def solve_shooting(
    v_func, E_guess, mass, x_grid, hbar=1.0, is_radial=False, parity="even"
):
    """
    Solve for the energy eigenvalue using the Shooting IVP Method.

    Parameters
    ----------
    v_func : callable
        Function returning the potential energy given a spatial coordinate.
    E_guess : float
        Initial guess for the energy eigenvalue.
    mass : float
        Mass of the particle.
    x_grid : numpy.ndarray
        The spatial grid array providing dx and L (limits).
    hbar : float, optional
        Reduced Planck constant. Default is 1.0.
    is_radial : bool, optional
        If True, adapts boundary conditions for radial coordinate $r$. Default is False.
    parity : str, optional
        Parity for the initial conditions ('even' or 'odd'). Ignored if `is_radial` is True.

    Returns
    -------
    float
        The refined energy eigenvalue found by the root scalar solver.
    """
    dx = x_grid[1] - x_grid[0]
    L = x_grid[-1]

    def schrodinger_ode(x_val, y, E):
        psi, dpsi = y
        V = v_func(x_val)
        d2psi = (2 * mass / hbar**2) * (V - E) * psi
        return [dpsi, d2psi]

    def shoot(E):
        if is_radial:
            y0 = [dx, 1.0]
            x0 = dx
        else:
            y0 = [1.0, 0.0] if parity == "even" else [0.0, 1.0]
            x0 = 0.0

        def divergence_event(x_val, y, E):
            return 1e50 - np.abs(y[0])

        divergence_event.terminal = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_ivp(
                schrodinger_ode,
                [x0, L],
                y0,
                args=(E,),
                rtol=1e-8,
                atol=1e-11,
                events=divergence_event,
            )
        return sol.y[0, -1]

    bracket_margin = max(0.1, abs(E_guess) * 0.05)
    root = root_scalar(
        shoot, bracket=[E_guess - bracket_margin, E_guess + bracket_margin]
    )
    return root.root
