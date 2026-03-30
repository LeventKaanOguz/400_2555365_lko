import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize, differential_evolution


def compute_expectation_value(ansatz, params, x_grid, v_grid, hbar, mass):
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
