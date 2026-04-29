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


def residuals(params, m_1, m_2, pdg_data, r):
    alpha_s, b, c = params
    sys = QuarkoniumSystem(m_1, m_2, alpha_s, b, c)

    # Solve states for each (L, S) channel (J-dependent terms are perturbative)
    evals_1S_0, _, evecs_1S_0, nu_1S_0 = solve_gem(sys, r, l=0, spin=0)
    evals_1S_1, _, evecs_1S_1, nu_1S_1 = solve_gem(sys, r, l=0, spin=1)
    evals_1P_0, _, evecs_1P_0, nu_1P_0 = solve_gem(sys, r, l=1, spin=0)
    evals_1P_1, _, evecs_1P_1, nu_1P_1 = solve_gem(sys, r, l=1, spin=1)  # For P-waves
    evals_1D_1, _, evecs_1D_1, nu_1D_1 = solve_gem(
        sys, r, l=2, spin=1
    )  # For D-waves (needed for mixing)

    # Calculate unmixed masses (without tensor shift for S=1, J=1 states)
    mass_1S_1_unmixed = get_mass(
        evals_1S_1,
        evecs_1S_1,
        nu_1S_1,
        sys,
        0,
        spin=1,
        l=0,
        j=1,
        include_tensor_shift=False,
    )
    mass_1D_1_unmixed = get_mass(
        evals_1D_1,
        evecs_1D_1,
        nu_1D_1,
        sys,
        0,
        spin=1,
        l=2,
        j=1,
        include_tensor_shift=False,
    )

    # Calculate diagonal tensor shifts for S=1, J=1 states
    tensor_shift_1S_1 = calc_tensor_shift_exact(
        evecs_1S_1[:, 0], nu_1S_1, sys, l=0, s=1, j=1
    )
    tensor_shift_1D_1 = calc_tensor_shift_exact(
        evecs_1D_1[:, 0], nu_1D_1, sys, l=2, s=1, j=1
    )

    # Calculate off-diagonal tensor mixing element <^3S_1 | V_T | ^3D_1>
    mixing_element = calc_tensor_mixing_exact(
        evecs_1S_1[:, 0], nu_1S_1, evecs_1D_1[:, 0], nu_1D_1, sys, s=1, j=1
    )

    # Construct and diagonalize the 2x2 mixing matrix for J=1, S=1 states
    mixing_matrix = np.array(
        [
            [mass_1S_1_unmixed + tensor_shift_1S_1, mixing_element],
            [mixing_element, mass_1D_1_unmixed + tensor_shift_1D_1],
        ]
    )
    mixed_masses = np.linalg.eigvalsh(mixing_matrix)
    # The lowest eigenvalue corresponds to the J/psi (1^3S_1) and the higher to psi(1^3D_1)
    mass_1S_1_mixed = mixed_masses[0]
    mass_1D_1_mixed = mixed_masses[1]

    calc = {
        # Ground states (n=1)
        "(1^1S)": get_mass(
            evals_1S_0,
            evecs_1S_0,
            nu_1S_0,
            sys,
            0,
            spin=0,
            l=0,
            j=0,
        ),
        "(1^3S)": mass_1S_1_mixed,  # Use mixed mass for J/psi
        "(1^1P)": get_mass(
            evals_1P_0,
            evecs_1P_0,
            nu_1P_0,
            sys,
            0,
            spin=0,
            l=1,
            j=1,
        ),
        "(1^3P_0)": get_mass(
            evals_1P_1,
            evecs_1P_1,
            nu_1P_1,
            sys,
            0,
            spin=1,
            l=1,
            j=0,
        ),
        "(1^3P_1)": get_mass(
            evals_1P_1,
            evecs_1P_1,
            nu_1P_1,
            sys,
            0,
            spin=1,
            l=1,
            j=1,
        ),
        "(1^3P_2)": get_mass(
            evals_1P_1,
            evecs_1P_1,
            nu_1P_1,
            sys,
            0,
            spin=1,
            l=1,
            j=2,
        ),
        "(1^3D_1)": mass_1D_1_mixed,  # Use mixed mass for psi(1D)
        # First excited states (n=2)
        "(2^1S)": get_mass(
            evals_1S_0,
            evecs_1S_0,
            nu_1S_0,
            sys,
            1,
            spin=0,
            l=0,
            j=0,
        ),
        "(2^3S)": get_mass(
            evals_1S_1,
            evecs_1S_1,
            nu_1S_1,
            sys,
            1,
            spin=1,
            l=0,
            j=1,
        ),  # This is for Y(2S)
        "(2^1P)": get_mass(
            evals_1P_0,
            evecs_1P_0,
            nu_1P_0,
            sys,
            1,
            spin=0,
            l=1,
            j=1,
        ),
        "(2^3P_0)": get_mass(
            evals_1P_1,
            evecs_1P_1,
            nu_1P_1,
            sys,
            1,
            spin=1,
            l=1,
            j=0,
        ),
        "(2^3P_1)": get_mass(
            evals_1P_1,
            evecs_1P_1,
            nu_1P_1,
            sys,
            1,
            spin=1,
            l=1,
            j=1,
        ),
        "(2^3P_2)": get_mass(
            evals_1P_1,
            evecs_1P_1,
            nu_1P_1,
            sys,
            1,
            spin=1,
            l=1,
            j=2,
        ),
    }

    # Apply custom weights. Ground S-wave states are the most phenomenologically
    # important to anchor. Excited and P-wave states can have a bit more slack.
    # Graded weights:
    # n=1 S-waves (Heaviest priority to anchor the potential)
    # n=2 S-waves (High priority)
    # n=1 P-waves (Medium priority)
    # Everything else defaults to 1.0 (Low priority)
    weights = {
        "(1^1S)": 4.0,  # eta_b / eta_c
        "(1^3S)": 4.0,  # Upsilon / J/psi
        "(2^1S)": 3.0,  # eta_b(2S) / eta_c(2S)
        "(2^3S)": 3.0,  # Upsilon(2S) / psi(2S)
        "(1^1P)": 2.0,  # h_b / h_c
        "(1^3P_0)": 2.0,  # chi_b0 / chi_c0
        "(1^3P_1)": 2.0,  # chi_b1 / chi_c1
        "(1^3P_2)": 2.0,  # chi_b2 / chi_c2
        "(1^3D_1)": 1.0,  # D-waves and higher states fallback to 1.0
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
            m_1,
            m_2,
            pdg_data,
            r,
            initial_guesses,
            output_csv=csv_path,
            bounds=bounds,
        )

    print(f"Loading cached parameters from {csv_path}")
    return load_parameters(csv_path)
