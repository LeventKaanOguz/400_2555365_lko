import csv
import os


def export_numerical_results(
    csv_path, methods_all, energies_e0, energies_e1, errors_e0, errors_e1
):
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
    print("--- Energy Eigenvalue Comparison Table ---")
    header = f"{'Calculation Method':<25} | {'E_0 (Ground State)':<20} | {'E_1 (1st Excited)'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for method, e0, e1 in zip(methods, e0_list, e1_list):
        print(f"{method:<25} | {e0:<20.6f} | {e1:.6f}")


def print_error_table(methods, err_e0_list, err_e1_list):
    print("\n--- Absolute Errors (vs Numerical Matrix FD) ---")
    error_header = f"{'Calculation Method':<25} | {'E_0 Error':<20} | {'E_1 Error'}"
    print("-" * len(error_header))
    print(error_header)
    print("-" * len(error_header))
    for method, err_e0, err_e1 in zip(methods, err_e0_list, err_e1_list):
        print(f"{method:<25} | {err_e0:<20.6e} | {err_e1:.6e}")


def format_poly_params(params, parity="even"):
    res = f"alpha={params[0]:.3f}"
    for i, c in enumerate(params[1:]):
        res += f", c_{2 * (i + 1) if parity == 'even' else 2 * (i + 1) + 1}={c:.3f}"
    return res
