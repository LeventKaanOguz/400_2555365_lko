import matplotlib.pyplot as plt
import numpy as np

# Import the runner and parameters from your existing code
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

    # --- Calculate Variational Wavefunctions (Probability Densities) ---
    # Ground state: Ansatz: (2alpha/pi)^1/4 * e^(-alpha x^2)
    psi0_var = (2 * alpha0 / np.pi) ** 0.25 * np.exp(-alpha0 * x**2)

    # First excited state: Ansatz: N * x * e^(-alpha x^2)
    N_var = np.sqrt(4 * alpha1 * np.sqrt(2 * alpha1 / np.pi))
    psi1_var = N_var * x * np.exp(-alpha1 * x**2)

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
        label="|ψ_0 Variational|²",
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
        label="|ψ_1 Variational|²",
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
    plt.show()


if __name__ == "__main__":
    create_showcase()
