import os
import csv
import numpy as np
from scipy.optimize import least_squares
from .gem_solver import QuarkoniumSystem, solve_gem
from .observables import (
    get_mass,
    calc_tensor_mixing_exact,
    calc_tensor_shift_exact,
)
from .decay_models import get_leptonic_width

# --- Physical error model ----------------------------------------------------
# The fit is a true (dimensionless) chi-square: every residual is divided by a
# physically motivated 1-sigma uncertainty rather than a hand-tuned weight.
#
# Experimental PDG mass errors are tiny (~0.1-1 MeV, see pdg_loader) -- far below
# the intrinsic accuracy of a quenched Cornell + perturbative-correction model.
# So the dominant, honest error bar on each predicted mass is a *theory
# systematic* SIGMA_THEORY_MEV (neglected relativistic O(v^2/c^2), coupled-channel
# and quenching effects). Using sigma_i = quad(sigma_exp, SIGMA_THEORY) makes the
# weighting essentially uniform without the precisely-measured states (J/psi,
# Upsilon) swamping everything, which is exactly what the old ad-hoc {4,3,2,1}
# weights were faking.
SIGMA_THEORY_MEV = 10.0
# Non-relativistic annihilation-width formulae (Gamma ~ |R(0)|^2) carry large
# perturbative-QCD and relativistic corrections; ~30% is a standard estimate.
WIDTH_THEORY_FRAC = 0.30
# Only sub-threshold, narrow S-wave vectors (n <= this) are clean enough to use
# as leptonic-width constraints in the fit.
DECAY_FIT_MAX_N = 2


def _solve_channels(params, m_1, m_2, r):
    """Solve the GEM eigensystem for every (L, S) channel the fit needs."""
    alpha_s, b, c, sigma = params
    sys = QuarkoniumSystem(m_1, m_2, alpha_s, b, c, sigma_smear=sigma)
    ch = {
        "1S0": solve_gem(sys, r, l=0, spin=0),
        "1S1": solve_gem(sys, r, l=0, spin=1),
        "1P0": solve_gem(sys, r, l=1, spin=0),
        "1P1": solve_gem(sys, r, l=1, spin=1),
        "1D1": solve_gem(sys, r, l=2, spin=1),
    }
    return sys, ch


def _masses_from_channels(sys, ch):
    """Build the model mass for every fitted state label from solved channels.

    This is the single source of truth for the predicted spectrum used in the
    fit; ``compute_spectrum_masses`` exposes it for cross-validation so the
    held-out evaluation uses exactly the same physics as the objective.
    """
    evals_1S_0, _, evecs_1S_0, nu_1S_0 = ch["1S0"]
    evals_1S_1, _, evecs_1S_1, nu_1S_1 = ch["1S1"]
    evals_1P_0, _, evecs_1P_0, nu_1P_0 = ch["1P0"]
    evals_1P_1, _, evecs_1P_1, nu_1P_1 = ch["1P1"]
    evals_1D_1, _, evecs_1D_1, nu_1D_1 = ch["1D1"]

    # Unmixed 2S and 1D (J=1, S=1) masses without the tensor shift
    mass_2S_1_unmixed = get_mass(
        evals_1S_1, evecs_1S_1, nu_1S_1, sys, 1, spin=1, l=0, j=1,
        include_tensor_shift=False,
    )
    mass_1D_1_unmixed = get_mass(
        evals_1D_1, evecs_1D_1, nu_1D_1, sys, 0, spin=1, l=2, j=1,
        include_tensor_shift=False,
    )
    tensor_shift_2S_1 = calc_tensor_shift_exact(
        evecs_1S_1[:, 1], nu_1S_1, sys, l=0, s=1, j=1
    )
    tensor_shift_1D_1 = calc_tensor_shift_exact(
        evecs_1D_1[:, 0], nu_1D_1, sys, l=2, s=1, j=1
    )
    mixing_element = calc_tensor_mixing_exact(
        evecs_1S_1[:, 1], nu_1S_1, evecs_1D_1[:, 0], nu_1D_1, sys, s=1, j=1
    )
    mixing_matrix = np.array(
        [
            [mass_2S_1_unmixed + tensor_shift_2S_1, mixing_element],
            [mixing_element, mass_1D_1_unmixed + tensor_shift_1D_1],
        ]
    )
    mixed_masses = np.linalg.eigvalsh(mixing_matrix)
    mass_2S_1_mixed = mixed_masses[0]
    mass_1D_1_mixed = mixed_masses[1]

    return {
        # Ground states (n=1)
        "(1^1S)": get_mass(evals_1S_0, evecs_1S_0, nu_1S_0, sys, 0, spin=0, l=0, j=0),
        "(1^3S)": get_mass(evals_1S_1, evecs_1S_1, nu_1S_1, sys, 0, spin=1, l=0, j=1),
        "(1^1P)": get_mass(evals_1P_0, evecs_1P_0, nu_1P_0, sys, 0, spin=0, l=1, j=1),
        "(1^3P_0)": get_mass(evals_1P_1, evecs_1P_1, nu_1P_1, sys, 0, spin=1, l=1, j=0),
        "(1^3P_1)": get_mass(evals_1P_1, evecs_1P_1, nu_1P_1, sys, 0, spin=1, l=1, j=1),
        "(1^3P_2)": get_mass(evals_1P_1, evecs_1P_1, nu_1P_1, sys, 0, spin=1, l=1, j=2),
        "(1^3D_1)": mass_1D_1_mixed,
        # First excited states (n=2)
        "(2^1S)": get_mass(evals_1S_0, evecs_1S_0, nu_1S_0, sys, 1, spin=0, l=0, j=0),
        "(2^3S)": mass_2S_1_mixed,
        "(2^1P)": get_mass(evals_1P_0, evecs_1P_0, nu_1P_0, sys, 1, spin=0, l=1, j=1),
        "(2^3P_0)": get_mass(evals_1P_1, evecs_1P_1, nu_1P_1, sys, 1, spin=1, l=1, j=0),
        "(2^3P_1)": get_mass(evals_1P_1, evecs_1P_1, nu_1P_1, sys, 1, spin=1, l=1, j=1),
        "(2^3P_2)": get_mass(evals_1P_1, evecs_1P_1, nu_1P_1, sys, 1, spin=1, l=1, j=2),
    }


def compute_spectrum_masses(params, m_1, m_2, r):
    """Model mass (GeV) for every fitted state label, as a dict. Shared by the
    fit residuals and the cross-validation evaluator."""
    sys, ch = _solve_channels(params, m_1, m_2, r)
    return _masses_from_channels(sys, ch)


def residuals(params, m_1, m_2, pdg_masses, pdg_mass_err, r, decay_targets=None, e_q=0.0):
    """Dimensionless (chi) residuals: (model - experiment) / sigma_physical.

    Mass residuals use sigma = quad(sigma_exp, SIGMA_THEORY_MEV). When
    ``decay_targets`` (a {state_label: Gamma_ee_keV} map) and a non-zero quark
    charge ``e_q`` are supplied, the measured S-wave leptonic widths are added as
    extra residuals weighted by WIDTH_THEORY_FRAC, so the wavefunction-at-origin
    helps constrain the potential -- not masses alone.
    """
    sys, ch = _solve_channels(params, m_1, m_2, r)
    calc = _masses_from_channels(sys, ch)

    res = []
    for state, exp_m in pdg_masses.items():
        if exp_m is None or state not in calc:
            continue
        sigma_exp = (pdg_mass_err or {}).get(state, 0.0) * 1000.0
        sigma = np.hypot(sigma_exp, SIGMA_THEORY_MEV)
        res.append((calc[state] - exp_m) * 1000.0 / sigma)

    if decay_targets and e_q != 0.0:
        evals_1S_1, _, evecs_1S_1, nu_1S_1 = ch["1S1"]
        for state, gamma_exp in decay_targets.items():
            if gamma_exp is None or gamma_exp <= 0 or not state.endswith("^3S)"):
                continue
            n = int(state[1])  # "(1^3S)" -> 1
            idx = n - 1
            if n > DECAY_FIT_MAX_N or idx >= len(evals_1S_1):
                continue
            gamma_calc = get_leptonic_width(
                calc[state], evecs_1S_1[:, idx], nu_1S_1, sys, e_q, l=0
            )
            sigma = WIDTH_THEORY_FRAC * gamma_exp
            res.append((gamma_calc - gamma_exp) / sigma)

    return res


def fit_and_save_parameters(
    m_1,
    m_2,
    pdg_data,
    r,
    initial_guesses,
    output_csv="results/fitted_parameters.csv",
    bounds=None,
    pdg_mass_err=None,
    decay_targets=None,
    e_q=0.0,
):
    """Run the optimization and save the fitted Cornell parameters + chi-square.

    Parameters
    ----------
    m_1, m_2 : float
        Quark masses (GeV).
    pdg_data : dict
        Experimental masses keyed by bare state label, e.g. ``{"(1^3S)": 9.4604}``.
    r : array-like
        Radial grid.
    initial_guesses : list
        Starting ``[alpha_s, b, c, sigma]``.
    output_csv : str
        Where to write the fitted parameters.
    bounds : tuple, optional
        ``(lower, upper)`` parameter bounds. A tight band freezes a parameter.
    pdg_mass_err : dict, optional
        Experimental 1-sigma mass errors (GeV) keyed by state label.
    decay_targets : dict, optional
        ``{state_label: Gamma_ee_keV}`` leptonic widths to fold into the fit.
    e_q : float, optional
        Quark electric charge (needed for the leptonic-width residuals).
    """
    print(f"Running fitter for {output_csv}... This may take a moment.")

    if bounds is None:
        # Phenomenological limits for [alpha_s, b, c, sigma] to prevent unphysical fits
        bounds = ([0.1, 0.1, -1.0, 0.3], [0.8, 0.35, 1.0, 5.0])

    # Linear loss -> the minimised objective is the true chi-square (residuals are
    # already normalised by physical sigmas, so robust losses are unnecessary and
    # would make the reported chi^2 non-standard).
    result = least_squares(
        residuals,
        initial_guesses,
        args=(m_1, m_2, pdg_data, pdg_mass_err, r, decay_targets, e_q),
        bounds=bounds,
        method="trf",  # handles bounds and the under-determined/frozen sectors
    )

    optimized_params = result.x

    # --- Goodness-of-fit (true reduced chi^2) of the optimisation step ---------
    # result.fun holds the dimensionless residuals (model - exp)/sigma, so the
    # plain chi^2 = sum_i residual_i^2 is a standard statistic. Count only the
    # genuinely free parameters: a tight bound (e.g. the frozen alpha_s/sigma of
    # B_c, or all but c for the D meson) is not a degree of freedom, so it must not
    # inflate the dof into a negative number.
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    n_free = int(np.sum((hi - lo) > 1e-4))
    residual_vec = np.asarray(result.fun)
    n_data = len(residual_vec)
    dof = n_data - n_free
    chi2_fit = float(np.sum(residual_vec**2))
    chi2_per_dof = chi2_fit / dof if dof > 0 else float("nan")
    print(
        f"  Fit chi^2 = {chi2_fit:.2f}  |  N_data = {n_data} "
        f"(masses + decays), N_free = {n_free}, dof = {dof}  |  "
        f"chi^2/dof = {chi2_per_dof:.3f}"
    )

    # Covariance and parameter errors from the Jacobian
    try:
        U, s, Vh = np.linalg.svd(result.jac, full_matrices=False)
        tol = np.finfo(float).eps * s[0] * max(result.jac.shape)
        w = s > tol
        cov = (Vh[w].T / s[w] ** 2) @ Vh[w]
        if dof > 0:
            s_sq = 2 * result.cost / dof
            cov *= s_sq
        else:
            cov = np.zeros((len(result.x), len(result.x)))
    except Exception:
        cov = np.zeros((len(result.x), len(result.x)))

    perr = np.sqrt(np.diag(cov))

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["alpha_s", "b", "c", "sigma", "err_alpha_s", "err_b", "err_c",
             "err_sigma", "chi2_fit", "dof", "chi2_per_dof"]
        )
        writer.writerow(
            list(optimized_params) + list(perr) + [chi2_fit, dof, chi2_per_dof]
        )

    print(f"Parameters successfully fitted and saved to {output_csv}")
    return optimized_params, perr


def load_parameters(csv_path="results/fitted_parameters.csv"):
    """Load fitted ``[alpha_s, b, c, sigma]`` and their errors from CSV."""
    with open(csv_path, mode="r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        row = next(reader)
        params = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
        perr = (
            [float(row[4]), float(row[5]), float(row[6]), float(row[7])]
            if len(row) > 4
            else [0.0, 0.0, 0.0, 0.0]
        )
        return params, perr


def get_or_fit_parameters(
    m_1,
    m_2,
    pdg_data,
    r,
    initial_guesses,
    csv_path="results/fitted_parameters.csv",
    force_refit=False,
    bounds=None,
    pdg_mass_err=None,
    decay_targets=None,
    e_q=0.0,
):
    """Load cached parameters if present, otherwise run (and cache) the fit."""
    if force_refit or not os.path.exists(csv_path):
        return fit_and_save_parameters(
            m_1, m_2, pdg_data, r, initial_guesses,
            output_csv=csv_path, bounds=bounds,
            pdg_mass_err=pdg_mass_err, decay_targets=decay_targets, e_q=e_q,
        )

    print(f"Loading cached parameters from {csv_path}")
    try:
        return load_parameters(csv_path)
    except (IndexError, ValueError):
        print(f"Old parameter format detected in {csv_path}, refitting...")
        return fit_and_save_parameters(
            m_1, m_2, pdg_data, r, initial_guesses,
            output_csv=csv_path, bounds=bounds,
            pdg_mass_err=pdg_mass_err, decay_targets=decay_targets, e_q=e_q,
        )
