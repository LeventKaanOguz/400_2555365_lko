import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_gem_coverage(
    csv_path, output_eps, state_col="c_1S", title_label="Bottomonium 1S"
):
    # Load the GEM coefficients
    df = pd.read_csv(csv_path)
    nu_n = df["nu_width"].values
    C_n = df[state_col].values

    # Generate r array on a logarithmic scale to show orders of magnitude
    # Range: ~0.01 to ~30 GeV^-1 (covers from well inside the cusp to the tail)
    r = np.logspace(-2, 1.5, 2000)

    # Normalization for S-wave (l=0) Gaussians
    # N_n = [ 4 * nu_n^(3/2) / sqrt(pi) ]^(1/2)
    N_n = np.sqrt(4 * nu_n ** (1.5) / np.sqrt(np.pi))

    plt.figure(figsize=(10, 6))

    total_u = np.zeros_like(r)

    # 1. Plot individual weighted basis functions: u_n(r) = C_n * r * N_n * exp(-nu_n * r^2)
    # We use a colormap to show the progression from sharpest to widest Gaussian
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(nu_n)))

    for i in range(len(nu_n)):
        # The radial probability amplitude u(r) = r * R(r)
        u_n = C_n[i] * r * N_n[i] * np.exp(-nu_n[i] * r**2)
        total_u += u_n

        plt.plot(r, u_n, "--", color=colors[i], alpha=0.6, linewidth=1.2)

    # 2. Plot the total physical wavefunction
    plt.plot(
        r,
        total_u,
        color="black",
        linewidth=2.5,
        label=f"Total Wavefunction $\\Psi_{{{title_label}}}(r)$",
    )

    # 3. Formatting the Plot (Standard Physical Review style)
    plt.xscale("log")
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)  # Zero line

    plt.title(f"GEM Basis Function Coverage: {title_label}", fontsize=14, pad=15)
    plt.xlabel(r"Radial distance $r$ [GeV$^{-1}$]", fontsize=12)
    plt.ylabel(r"Radial Amplitude $u(r) = r \cdot R(r)$", fontsize=12)

    plt.legend(loc="best", fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # Optional: Set x-limits tightly around the physical region
    plt.xlim(1e-2, 30)

    # 4. Save as EPS
    plt.savefig(output_eps, format="eps", bbox_inches="tight")
    print(f"Success! Plot saved to: {output_eps}")


if __name__ == "__main__":
    # Define your paths (adjust if your working directory is different)
    input_csv = "results/Bottomonium (b_bbar)_S_Wave_GEM_Coefficients.csv"
    output_file = "results/bottomonium_1S_gem_coverage.eps"

    if os.path.exists(input_csv):
        plot_gem_coverage(
            input_csv, output_file, state_col="c_1S", title_label="Bottomonium 1S"
        )
    else:
        print(
            f"Error: Could not find {input_csv}. Ensure you are running this from the project root."
        )
