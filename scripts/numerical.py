import warnings
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.integrate import simpson


def normalize_wavefunction(psi_unnorm, x_grid):
    norm = np.sqrt(simpson(y=psi_unnorm**2, x=x_grid))
    return psi_unnorm / norm


def solve_fd(x_grid, v_grid, hbar, mass):
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
