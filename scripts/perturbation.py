import numpy as np
from scipy.integrate import simpson, quad


def calc_analytical_pert_energy(
    n, v_pert_func, unperturbed_state_func, e_0_func, limit, max_states=10
):
    """
    Calculate 1st and 2nd order perturbation energy corrections using analytical integration.

    Parameters
    ----------
    n : int
        The quantum number of the state to correct.
    v_pert_func : callable
        The perturbing potential function, V'(x).
    unperturbed_state_func : callable
        Function returning the unperturbed wavefunction for a given state index and x.
    e_0_func : callable
        Function returning the unperturbed energy for a given state index.
    limit : float
        The absolute integration limits (-limit to limit) for `scipy.integrate.quad`.
    max_states : int, optional
        Maximum number of states to sum over for the 2nd order correction. Default is 10.

    Returns
    -------
    tuple
        (e_1st_order, e_2nd_order): Energies evaluated at the respective perturbation orders.
    """

    def integrand_1st(x_val):
        psi = unperturbed_state_func(n, x_val)
        return psi**2 * v_pert_func(x_val)

    e_1, _ = quad(integrand_1st, -limit, limit)

    e_2 = 0.0
    e_0 = e_0_func(n)
    for k in range(max_states):
        if k == n:
            continue

        def integrand_2nd(x_val):
            return (
                unperturbed_state_func(k, x_val)
                * v_pert_func(x_val)
                * unperturbed_state_func(n, x_val)
            )

        h_kn, _ = quad(integrand_2nd, -limit, limit)
        if abs(h_kn) > 1e-10:
            e_k = e_0_func(k)
            e_2 += (h_kn**2) / (e_0 - e_k)

    return e_0 + e_1, e_0 + e_1 + e_2


def calc_numerical_pert_energy(
    n, x_grid, v_pert_grid, evals_unperturbed, psi_unperturbed, max_states=15
):
    """
    Calculate 1st and 2nd order perturbation energy corrections using numerical integration.

    Parameters
    ----------
    n : int
        The quantum number of the state to correct.
    x_grid : numpy.ndarray
        The spatial grid array.
    v_pert_grid : numpy.ndarray
        The perturbing potential evaluated over the spatial grid.
    evals_unperturbed : numpy.ndarray
        Array of unperturbed energy eigenvalues.
    psi_unperturbed : numpy.ndarray
        2D array where columns correspond to unperturbed eigenfunctions.
    max_states : int, optional
        Maximum number of states to sum over for the 2nd order correction. Default is 15.

    Returns
    -------
    tuple
        (e_1st_order, e_2nd_order): Energies evaluated at the respective perturbation orders.
    """
    e_0 = evals_unperturbed[n]
    psi_n = psi_unperturbed[:, n]

    e_1 = simpson(y=psi_n**2 * v_pert_grid, x=x_grid)

    e_2 = 0.0
    for k in range(max_states):
        if k == n:
            continue
        h_kn = simpson(y=psi_unperturbed[:, k] * v_pert_grid * psi_n, x=x_grid)

        if abs(h_kn) > 1e-10:
            e_2 += h_kn**2 / (e_0 - evals_unperturbed[k])

    return e_0 + e_1, e_0 + e_1 + e_2
