import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import csv
from scipy.integrate import simpson
from qho_osc import run_comparisons, SIGMA


def create_showcase():
    print("Gathering data from qho_osc.py...")
    results = run_comparisons()

    x = results["x"]
    v_total = results["v_total"]
    e0_fd = results["e0_fd"]
    e1_fd = results["e1_fd"]
    psi0_fd = results["psi0_fd"]
    psi1_fd = results["psi1_fd"]
    alpha0 = results["alpha0"]
    alpha1 = results["alpha1"]
    alpha0_exp = results["alpha0_exp"]
    alpha1_exp = results["alpha1_exp"]
    params0_poly = results["params0_poly"]
    params1_poly = results["params1_poly"]
    params0_poly_gauss = results["params0_poly_gauss"]
    params1_poly_gauss = results["params1_poly_gauss"]

    # --- Calculate Variational Wavefunctions (Probability Densities) ---
    # Ground state: Ansatz: (2alpha/pi)^1/4 * e^(-alpha x^2)
    psi0_var = (2 * alpha0 / np.pi) ** 0.25 * np.exp(-alpha0 * x**2)

    # First excited state: Ansatz: N * x * e^(-alpha x^2)
    N_var = np.sqrt(4 * alpha1 * np.sqrt(2 * alpha1 / np.pi))
    psi1_var = N_var * x * np.exp(-alpha1 * x**2)

    # Exponential Ansatzes
    # Ground state: N * e^(-alpha |x|) -> N = sqrt(alpha)
    psi0_exp = np.sqrt(alpha0_exp) * np.exp(-alpha0_exp * np.abs(x))
    # First excited state: N * x * e^(-alpha |x|) -> N = sqrt(2 * alpha^3)
    psi1_exp = np.sqrt(2 * alpha1_exp**3) * x * np.exp(-alpha1_exp * np.abs(x))

    # Helper functions for arbitrary power series evaluation
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

    # Polynomial-Exponential Ansatzes (Numerical Normalization)
    psi0_poly_unnorm = np.exp(-params0_poly[0] * np.abs(x)) * build_even_poly(
        x, params0_poly[1:]
    )
    norm0_poly = np.sqrt(simpson(y=psi0_poly_unnorm**2, x=x))
    psi0_poly = psi0_poly_unnorm / norm0_poly

    psi1_poly_unnorm = np.exp(-params1_poly[0] * np.abs(x)) * build_odd_poly(
        x, params1_poly[1:]
    )
    norm1_poly = np.sqrt(simpson(y=psi1_poly_unnorm**2, x=x))
    psi1_poly = psi1_poly_unnorm / norm1_poly

    # Polynomial-Gaussian Ansatzes (Numerical Normalization)
    psi0_poly_gauss_unnorm = np.exp(-params0_poly_gauss[0] * x**2) * build_even_poly(
        x, params0_poly_gauss[1:]
    )
    norm0_poly_gauss = np.sqrt(simpson(y=psi0_poly_gauss_unnorm**2, x=x))
    psi0_poly_gauss = psi0_poly_gauss_unnorm / norm0_poly_gauss

    psi1_poly_gauss_unnorm = np.exp(-params1_poly_gauss[0] * x**2) * build_odd_poly(
        x, params1_poly_gauss[1:]
    )
    norm1_poly_gauss = np.sqrt(simpson(y=psi1_poly_gauss_unnorm**2, x=x))
    psi1_poly_gauss = psi1_poly_gauss_unnorm / norm1_poly_gauss

    # --- Setup the Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Quantum Harmonic Oscillator Showcase (σ = {SIGMA})", fontsize=16)

    # ==========================================
    # SUBPLOT 1: Potential and Wavefunctions
    # ==========================================
    # Plot the potential V(x)
    ax1.plot(x, v_total, color="black", lw=2, label="V(x) = 1/2 mω²x² + σx⁴")

    # Plot Energy levels
    ax1.axhline(
        e0_fd, color="blue", linestyle="--", alpha=0.5, label=f"E_0 = {e0_fd:.3f}"
    )
    ax1.axhline(
        e1_fd, color="red", linestyle="--", alpha=0.5, label=f"E_1 = {e1_fd:.3f}"
    )

    # Plot probability densities scaled and shifted up by their respective energy levels
    scale = 0.5  # scaling factor purely for visual clarity against the potential curve
    ax1.fill_between(
        x, e0_fd, e0_fd + scale * psi0_fd**2, color="blue", alpha=0.3, label="|ψ_0 FD|²"
    )
    ax1.plot(
        x,
        e0_fd + scale * psi0_var**2,
        color="cyan",
        linestyle=":",
        lw=2.5,
        label="|ψ_0 Var (Gauss)|²",
    )
    ax1.plot(
        x,
        e0_fd + scale * psi0_exp**2,
        color="green",
        linestyle="-.",
        lw=2.0,
        label="|ψ_0 Var (Exp)|²",
    )
    ax1.plot(
        x,
        e0_fd + scale * psi0_poly**2,
        color="purple",
        linestyle="--",
        lw=2.0,
        label="|ψ_0 Var (Poly-Exp)|²",
    )
    ax1.plot(
        x,
        e0_fd + scale * psi0_poly_gauss**2,
        color="olive",
        linestyle="-.",
        lw=2.0,
        label="|ψ_0 Var (Poly-Gauss)|²",
    )

    ax1.fill_between(
        x, e1_fd, e1_fd + scale * psi1_fd**2, color="red", alpha=0.3, label="|ψ_1 FD|²"
    )
    ax1.plot(
        x,
        e1_fd + scale * psi1_var**2,
        color="orange",
        linestyle=":",
        lw=2.5,
        label="|ψ_1 Var (Gauss)|²",
    )
    ax1.plot(
        x,
        e1_fd + scale * psi1_exp**2,
        color="magenta",
        linestyle="-.",
        lw=2.0,
        label="|ψ_1 Var (Exp)|²",
    )
    ax1.plot(
        x,
        e1_fd + scale * psi1_poly**2,
        color="brown",
        linestyle="--",
        lw=2.0,
        label="|ψ_1 Var (Poly-Exp)|²",
    )
    ax1.plot(
        x,
        e1_fd + scale * psi1_poly_gauss**2,
        color="navy",
        linestyle="-.",
        lw=2.0,
        label="|ψ_1 Var (Poly-Gauss)|²",
    )

    # Zoom in to the relevant part of the potential well
    display_limit = 4.0 / (1.0 + SIGMA) ** 0.25
    ax1.set_xlim(-display_limit, display_limit)
    ax1.set_ylim(0, e1_fd + 1.5)
    ax1.set_title("Wavefunction Probability Densities vs Potential")
    ax1.set_xlabel("Position (x)")
    ax1.set_ylabel("Energy")
    ax1.legend(loc="upper right", fontsize="small")

    # ==========================================
    # SUBPLOT 2: Error Analysis Bar Chart
    # ==========================================
    x_pos = np.arange(len(results["methods"]))
    width = 0.35

    ax2.bar(
        x_pos - width / 2,
        results["errors_e0"],
        width,
        label="E_0 Error",
        color="blue",
        alpha=0.7,
    )
    ax2.bar(
        x_pos + width / 2,
        results["errors_e1"],
        width,
        label="E_1 Error",
        color="red",
        alpha=0.7,
    )

    ax2.set_yscale("log")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results["methods"], rotation=15, ha="right")
    ax2.set_title("Absolute Energy Errors vs Numerical FD (Log Scale)")
    ax2.set_ylabel("Absolute Error (Log Scale)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/figures/qho_showcase.png", dpi=300)
    print("\nShowcase image successfully generated and saved as 'qho_showcase.png'")

    # ==========================================
    # EXPORT RESULTS TO CSV TABLE
    # ==========================================
    os.makedirs("results/tables", exist_ok=True)
    csv_path = "results/tables/qho_numerical_results.csv"

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

        for i, method in enumerate(results["methods_all"]):
            e0_val = results["energies_e0"][i]
            e1_val = results["energies_e1"][i]

            if method == "Numerical: Matrix FD":
                str_err_e0, str_err_e1 = "-", "-"
            else:
                str_err_e0 = f"{results['errors_e0'][i]:.2e}"
                str_err_e1 = f"{results['errors_e1'][i]:.2e}"

            writer.writerow(
                [
                    method,
                    f"{e0_val:.5f}",
                    f"{e1_val:.5f}",
                    str_err_e0,
                    str_err_e1,
                ]
            )

    print(f"Numerical results table successfully generated and saved as '{csv_path}'")
    plt.show()


if __name__ == "__main__":
    create_showcase()
