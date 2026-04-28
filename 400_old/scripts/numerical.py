import warnings
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.integrate import simpson


def normalize_wavefunction(psi_unnorm, x_grid):
    """
    Normalize a given wavefunction using Simpson's rule.

    Parameters
    ----------
    psi_unnorm : numpy.ndarray
        The unnormalized wavefunction array.
    x_grid : numpy.ndarray
        The spatial grid array.

    Returns
    -------
    numpy.ndarray
        The normalized wavefunction array.
    """
    norm = np.sqrt(simpson(y=psi_unnorm**2, x=x_grid))
    return psi_unnorm / norm


def solve_fd(x_grid, v_grid, hbar, mass):
    """
    Solve the 1D Schrödinger equation using the Finite Difference (FD) method.

    This method discretizes the kinetic energy operator into a sparse
    tridiagonal matrix and directly solves the resulting eigenvalue problem.

    Parameters
    ----------
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
    tuple
        A tuple containing:
        - evals (numpy.ndarray): The calculated energy eigenvalues.
        - psi_fd (numpy.ndarray): The corresponding normalized eigenfunctions.
    """
    dx = x_grid[1] - x_grid[0]
    N = len(x_grid)
    diag_kin = hbar**2 / (mass * dx**2)
    off_diag_kin = -(hbar**2) / (2 * mass * dx**2)

    main_diag = diag_kin + v_grid
    off_diag = np.full(N - 1, off_diag_kin)

    evals, evecs = eigh_tridiagonal(main_diag, off_diag)

    psi_fd = evecs / np.sqrt(dx)
    return evals, psi_fd


def solve_shooting(
    v_func, E_guess, parity, L, hbar, mass, bracket_margin=0.1, is_radial=False, x0=0.0
):
    """
    Solve for the energy eigenvalue using the Shooting Method (IVP).

    Integrates the Schrödinger equation outward from `x0` to `L`. Finds the energy
    where the boundary condition at infinity (wavefunction approaches zero) is met.

    Parameters
    ----------
    v_func : callable
        A function taking a spatial coordinate `x` and returning potential energy.
    E_guess : float
        Initial guess for the energy eigenvalue (e.g., from the FD method).
    parity : str
        The parity of the state ('even' or 'odd'). Ignored if `is_radial=True`.
    L : float
        The spatial boundary (infinity approximation) where integration stops.
    hbar : float
        Reduced Planck constant.
    mass : float
        Mass of the particle (or reduced mass of the system).
    bracket_margin : float, optional
        Margin around the `E_guess` to search for the root. Default is 0.1.
    is_radial : bool, optional
        If True, sets the initial boundary conditions for a 3D radial equation. Default is False.
    x0 : float, optional
        The starting coordinate for integration. Default is 0.0.

    Returns
    -------
    float
        The refined energy eigenvalue found by the root scalar solver.
    """

    def schrodinger_ode(x_val, y, E):
        psi, dpsi = y
        V = v_func(x_val)
        d2psi = (2 * mass / hbar**2) * (V - E) * psi
        return [dpsi, d2psi]

    def shoot(E, parity):
        if is_radial:
            y0 = [x0, 1.0]
        else:
            y0 = [1.0, 0.0] if parity == "even" else [0.0, 1.0]

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

    root = root_scalar(
        shoot,
        args=(parity,),
        bracket=[E_guess - bracket_margin, E_guess + bracket_margin],
    )
    return root.root
