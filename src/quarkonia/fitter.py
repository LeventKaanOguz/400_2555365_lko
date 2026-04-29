import os
import csv
from scipy.optimize import least_squares
from .gem_solver import QuarkoniumSystem, solve_gem
from .observables import get_mass


def residuals(params, m_1, m_2, pdg_data, r):
    alpha_s, b, c = params
    sys = QuarkoniumSystem(m_1, m_2, alpha_s, b, c)

    evals_s, u_s, _, _ = solve_gem(sys, r, l=0)
    evals_p, u_p, _, _ = solve_gem(sys, r, l=1)

    calc = {
        # Ground states (n=1)
        "(1^1S)": get_mass(evals_s, u_s, r, sys, 0, spin=0, l=0),
        "(1^3S)": get_mass(evals_s, u_s, r, sys, 0, spin=1, l=0),
        "(1^1P)": get_mass(evals_p, u_p, r, sys, 0, spin=0, l=1, j=1),
        "(1^3P_0)": get_mass(evals_p, u_p, r, sys, 0, spin=1, l=1, j=0),
        "(1^3P_1)": get_mass(evals_p, u_p, r, sys, 0, spin=1, l=1, j=1),
        "(1^3P_2)": get_mass(evals_p, u_p, r, sys, 0, spin=1, l=1, j=2),
        # First excited states (n=2)
        "(2^1S)": get_mass(evals_s, u_s, r, sys, 1, spin=0, l=0),
        "(2^3S)": get_mass(evals_s, u_s, r, sys, 1, spin=1, l=0),
        "(2^1P)": get_mass(evals_p, u_p, r, sys, 1, spin=0, l=1, j=1),
        "(2^3P_0)": get_mass(evals_p, u_p, r, sys, 1, spin=1, l=1, j=0),
        "(2^3P_1)": get_mass(evals_p, u_p, r, sys, 1, spin=1, l=1, j=1),
        "(2^3P_2)": get_mass(evals_p, u_p, r, sys, 1, spin=1, l=1, j=2),
    }

    # Apply custom weights. Ground S-wave states are the most phenomenologically
    # important to anchor. Excited and P-wave states can have a bit more slack.
    weights = {
        "(1^1S)": 5.0,
        "(1^3S)": 5.0,
        "(2^1S)": 2.0,
        "(2^3S)": 2.0,
    }

    return [
        (calc[state] - exp_m) * 1000.0 * weights.get(state, 1.0)
        for state, exp_m in pdg_data.items()
        if exp_m is not None and state in calc
    ]


def fit_and_save_parameters(
    m_1,
    m_2,
    pdg_data,
    r,
    initial_guesses,
    output_csv="results/fitted_parameters.csv",
    bounds=None,
):
    """Runs the optimization and saves the result to a CSV file."""
    print(f"Running fitter for {output_csv}... This may take a moment.")

    if bounds is None:
        # Phenomenological limits for [alpha_s, b, c] to prevent unphysical fits
        bounds = ([0.1, 0.1, -1.0], [0.8, 0.35, 1.0])

    # Run the least squares optimization
    result = least_squares(
        residuals,
        initial_guesses,
        args=(m_1, m_2, pdg_data, r),
        bounds=bounds,
        method="trf",  # 'trf' handles bounds and underdetermined systems (like the B_c sector)
        loss="soft_l1",  # Robust loss to prevent outlier states from skewing the global fit
    )

    optimized_params = result.x

    # Ensure the directory exists and save parameters to a CSV file
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha_s", "b", "c"])
        writer.writerow(optimized_params)

    print(f"Parameters successfully fitted and saved to {output_csv}")
    return optimized_params


def load_parameters(csv_path="results/fitted_parameters.csv"):
    """Loads fitted parameters from a CSV file."""
    with open(csv_path, mode="r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [float(val) for val in next(reader)]


def get_or_fit_parameters(
    m_1,
    m_2,
    pdg_data,
    r,
    initial_guesses,
    csv_path="results/fitted_parameters.csv",
    force_refit=False,
    bounds=None,
):
    """Automatically runs the fitter if the CSV is missing, otherwise loads from CSV."""
    if force_refit or not os.path.exists(csv_path):
        return fit_and_save_parameters(
            m_1, m_2, pdg_data, r, initial_guesses, output_csv=csv_path, bounds=bounds
        )

    print(f"Loading cached parameters from {csv_path}")
    return load_parameters(csv_path)
