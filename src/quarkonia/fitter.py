import os
import csv
import numpy as np
from scipy.optimize import least_squares
from .gem_solver import QuarkoniumSystem, solve_gem
from .observables import (
    get_mass,
    calc_tensor_mixing_exact,
    calc_tensor_shift_exact,
    calc_relativistic_shift,
)
from .decay_models import get_leptonic_width, get_running_alpha_s

# --- Physical error model (derived, not tuned) -------------------------------
# The fit is a true (dimensionless) chi-square: every residual is divided by a
# physically motivated 1-sigma uncertainty *computed from the model*, not a
# hand-tuned floor.
#
# Experimental PDG mass errors are tiny (~0.1-1 MeV, see pdg_loader) -- far below
# the intrinsic accuracy of a non-relativistic Cornell model. The honest error
# bar on each predicted mass is therefore a *theory* uncertainty: the magnitude
# of the leading omitted physics. For a non-relativistic potential model that is
# the leading relativistic correction, Delta H = -p^4/(8c^2)(1/m1^3 + 1/m2^3),
# the next term in the v^2/c^2 expansion. Its expectation value on each state
# (observables.calc_relativistic_shift) is the "size of the first omitted term"
# -- the standard effective-theory truncation error -- and is used as the
# per-state mass sigma. It is sector- AND state-dependent with zero free
# parameters: charm (<v^2> ~ 0.24) automatically gets a larger bar than bottom
# (<v^2> ~ 0.085), and the precisely-measured J/psi / Upsilon no longer swamp the
# fit. (This replaces the old hand-tuned flat 10 MeV floor.)
#
# The annihilation-width formulae (Gamma ~ |R(0)|^2) carry an O(alpha_s) QCD
# radiative correction (the 16 alpha_s/3pi factor already in get_leptonic_width)
# plus O(v^2) relativistic corrections. The per-sector fractional width
# uncertainty is the size of those, quad(16 alpha_s/3pi, sqrt(<v^2>)) -- ~0.6 for
# bottom, ~0.9 for charm -- again derived, replacing the old flat 30%.

# Only sub-threshold, narrow S-wave vectors (n <= this) are clean enough to use
# as leptonic-width constraints in the fit.
DECAY_FIT_MAX_N = 2

# Map a bare state label "(n^{2S+1}L...)" to the solved (L,S) channel key and the
# radial index n-1, so the per-state relativistic sigma can be read off the same
# eigenvectors the masses come from.
_L_OF = {"S": 0, "P": 1, "D": 2, "F": 3}


def _state_channel(state):
    """(channel_key, n_index, l) for a bare state label, or None if not solved."""
    try:
        n = int(state[1])
        spin = (int(state[3]) - 1) // 2  # 2S+1 -> S
        lchar = state[4]
    except (IndexError, ValueError):
        return None
    if lchar not in _L_OF:
        return None
    return f"1{lchar}{spin}", n - 1, _L_OF[lchar]


def sigma_theory_gev(ch, sys, state):
    """Per-state mass theory sigma (GeV) = |leading relativistic correction|.

    ``ch`` is the dict of solved channels (key -> (evals, u, evecs, nu)) built by
    :func:`_solve_channels`. Returns ``None`` when the state has no solved channel.
    """
    info = _state_channel(state)
    if info is None:
        return None
    key, idx, l = info
    if key not in ch:
        return None
    _, _, evecs, nu = ch[key]
    if idx < 0 or idx >= evecs.shape[1]:
        return None
    dE_rel, _ = calc_relativistic_shift(evecs[:, idx], nu, sys, l)
    return dE_rel


def width_theory_frac(ch, sys):
    """Per-sector fractional width uncertainty = quad(16 alpha_s/3pi, sqrt(<v^2>)).

    The O(alpha_s) QCD radiative factor (same coupling get_leptonic_width applies)
    and the O(v^2) relativistic scale, both derived. ``<v^2>`` is taken from the
    1^3S_1 ground state.
    """
    alpha_s_run = get_running_alpha_s(sys)
    qcd_mag = 16.0 * alpha_s_run / (3.0 * np.pi)
    _, _, evecs_1S1, nu_1S1 = ch["1S1"]
    _, v2 = calc_relativistic_shift(evecs_1S1[:, 0], nu_1S1, sys, l=0)
    return float(np.hypot(qcd_mag, np.sqrt(v2)))


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


def residuals(
    params, m_1, m_2, pdg_masses, pdg_mass_err, r,
    decay_targets=None, e_q=0.0, sigma_mode="state",
):
    """Dimensionless (chi) residuals: (model - experiment) / sigma_physical.

    Mass residuals use ``sigma = quad(sigma_exp, sigma_theory)`` where
    ``sigma_theory`` is the derived leading relativistic correction
    (:func:`sigma_theory_gev`). With ``sigma_mode="state"`` (default) it is the
    state's own |dE_rel|; with ``sigma_mode="sector"`` it is one value for the
    whole sector (the mean over the fitted states), the only sanctioned fallback
    if the self-consistent per-state fit fails to converge.

    When ``decay_targets`` ({state_label: Gamma_ee_keV}) and a non-zero quark
    charge ``e_q`` are supplied, the measured S-wave leptonic widths are added as
    extra residuals weighted by the derived per-sector fraction
    (:func:`width_theory_frac`), so the wavefunction-at-origin helps constrain the
    potential -- not masses alone.
    """
    sys, ch = _solve_channels(params, m_1, m_2, r)
    calc = _masses_from_channels(sys, ch)

    # Per-state theory sigma (MeV) for every fitted state, derived from the same
    # eigenvectors the masses come from.
    sig_theory = {}
    for state in pdg_masses:
        if pdg_masses[state] is None or state not in calc:
            continue
        s = sigma_theory_gev(ch, sys, state)
        if s is not None:
            sig_theory[state] = s * 1000.0  # GeV -> MeV
    sector_sigma = (
        float(np.mean(list(sig_theory.values()))) if sig_theory else 10.0
    )

    res = []
    for state, exp_m in pdg_masses.items():
        if exp_m is None or state not in calc:
            continue
        sigma_exp = (pdg_mass_err or {}).get(state, 0.0) * 1000.0
        if sigma_mode == "sector":
            sigma_th = sector_sigma
        else:
            sigma_th = sig_theory.get(state, sector_sigma)
        sigma = np.hypot(sigma_exp, sigma_th)
        res.append((calc[state] - exp_m) * 1000.0 / sigma)

    if decay_targets and e_q != 0.0:
        frac = width_theory_frac(ch, sys)
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
            sigma = frac * gamma_exp
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
    #
    # The physical sigma is the derived per-state relativistic correction
    # (sigma_mode="state"). This is self-consistent (the weights depend on the
    # current wavefunctions), so it is iteratively re-evaluated by least_squares.
    # The ONLY sanctioned fallback, if the per-state fit fails to converge, is a
    # single derived per-sector sigma -- never a frozen/initial-guess sigma.
    def _fit(mode):
        return least_squares(
            residuals,
            initial_guesses,
            args=(m_1, m_2, pdg_data, pdg_mass_err, r, decay_targets, e_q, mode),
            bounds=bounds,
            method="trf",  # handles bounds and the under-determined/frozen sectors
        )

    result = _fit("state")
    if not result.success:
        print(
            "  [warn] per-state sigma fit did not converge "
            f"(status={result.status}); retrying with a derived per-sector sigma."
        )
        result = _fit("sector")

    optimized_params = result.x

    # --- Goodness-of-fit (true reduced chi^2) of the optimisation step ---------
    # result.fun holds the dimensionless residuals (model - exp)/sigma, so the
    # plain chi^2 = sum_i residual_i^2 is a standard statistic. Count only the
    # genuinely free parameters: a tight bound (e.g. the frozen alpha_s/sigma of
    # B_c) is not a degree of freedom, so it must not inflate the dof into a
    # negative number.
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

    # Derived per-sector fractional width uncertainty (for the width pulls in the
    # consolidated report), computed once from the converged wavefunctions.
    try:
        sys_final, ch_final = _solve_channels(optimized_params, m_1, m_2, r)
        wfrac = width_theory_frac(ch_final, sys_final)
    except Exception:
        wfrac = float("nan")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["alpha_s", "b", "c", "sigma", "err_alpha_s", "err_b", "err_c",
             "err_sigma", "chi2_fit", "dof", "chi2_per_dof", "width_frac"]
        )
        writer.writerow(
            list(optimized_params) + list(perr)
            + [chi2_fit, dof, chi2_per_dof, wfrac]
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
