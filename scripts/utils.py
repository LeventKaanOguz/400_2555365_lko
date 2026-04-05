import csv
import os


def export_numerical_results(
    csv_path, methods_all, energies_e0, energies_e1, errors_e0, errors_e1
):
    """
    Export calculated energies and corresponding errors to a CSV file.

    Parameters
    ----------
    csv_path : str
        The destination file path for the CSV output.
    methods_all : list of str
        List containing names of the calculation methods used.
    energies_e0 : list of float
        Calculated ground state energies for each method.
    energies_e1 : list of float
        Calculated first excited state energies for each method.
    errors_e0 : list of float
        Absolute errors of the ground state energies.
    errors_e1 : list of float
        Absolute errors of the first excited state energies.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Method",
                "E0 (Ground State)",
                "E1 (1st Excited)",
                "E0 Absolute Error",
                "E1 Absolute Error",
            ]
        )

        for i, method in enumerate(methods_all):
            e0_val = energies_e0[i]
            e1_val = energies_e1[i]

            if method == "Numerical: Matrix FD":
                str_err_e0, str_err_e1 = "-", "-"
            else:
                str_err_e0 = f"{errors_e0[i]:.2e}"
                str_err_e1 = f"{errors_e1[i]:.2e}"

            writer.writerow(
                [method, f"{e0_val:.5f}", f"{e1_val:.5f}", str_err_e0, str_err_e1]
            )


def print_energy_table(methods, e0_list, e1_list):
    """
    Print a formatted console table comparing energy eigenvalues across methods.

    Parameters
    ----------
    methods : list of str
        List containing names of the calculation methods.
    e0_list : list of float
        Calculated ground state energies.
    e1_list : list of float
        Calculated first excited state energies.
    """
    print("--- Energy Eigenvalue Comparison Table ---")
    header = f"{'Calculation Method':<25} | {'E_0 (Ground State)':<20} | {'E_1 (1st Excited)'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for method, e0, e1 in zip(methods, e0_list, e1_list):
        print(f"{method:<25} | {e0:<20.6f} | {e1:.6f}")


def print_error_table(methods, err_e0_list, err_e1_list):
    """
    Print a formatted console table showing absolute energy errors.

    Parameters
    ----------
    methods : list of str
        List containing names of the calculation methods.
    err_e0_list : list of float
        Absolute errors corresponding to the ground state energies.
    err_e1_list : list of float
        Absolute errors corresponding to the first excited state energies.
    """
    print("\n--- Absolute Errors (vs Numerical Matrix FD) ---")
    error_header = f"{'Calculation Method':<25} | {'E_0 Error':<20} | {'E_1 Error'}"
    print("-" * len(error_header))
    print(error_header)
    print("-" * len(error_header))
    for method, err_e0, err_e1 in zip(methods, err_e0_list, err_e1_list):
        print(f"{method:<25} | {err_e0:<20.6e} | {err_e1:.6e}")


def format_poly_params(params, parity="even"):
    """
    Format optimized polynomial array parameters into a human-readable string.

    Parameters
    ----------
    params : list or numpy.ndarray
        List of parameters where index 0 is the exponent term alpha,
        and subsequent indices are the polynomial coefficient extensions.
    parity : str, optional
        'even' for evenly-powered polynomial terms, 'odd' for odd-powered. Default is 'even'.

    Returns
    -------
    str
        A formatted string describing the optimized parameters.
    """
    res = f"alpha={params[0]:.3f}"
    for i, c in enumerate(params[1:]):
        res += f", c_{2 * (i + 1) if parity == 'even' else 2 * (i + 1) + 1}={c:.3f}"
    return res
