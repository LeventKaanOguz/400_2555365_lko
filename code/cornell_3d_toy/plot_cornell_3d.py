import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from cornell_3d_toy import run_comparisons, ALPHA_S, B, C


def create_showcase():
    print("Gathering data from cornell_3d_toy.py...")
    results = run_comparisons()

    r = results["r"]
    v_total = results["v_total"]

    # S-Wave States
    e1s_fd = results["e1s_fd"]
    e2s_fd = results["e2s_fd"]
    u1s_fd = results["u1s_fd"]
    u2s_fd = results["u2s_fd"]
    u1s_gem = results["u1s_gem"]
    u2s_gem = results["u2s_gem"]

    # Orbital Excitations (P and D waves)
    e1p_fd_bare = results["evals_fd_1"][0]
    u1p_fd = results["u_fd_1"][:, 0]
    e1d_fd_bare = results["evals_fd_2"][0]
    u1d_fd = results["u_fd_2"][:, 0]

    # --- Setup Output Directory ---
    fig_dir = "results/figures/cornell_3d_toy"
    os.makedirs(fig_dir, exist_ok=True)

    scale = 0.6  # Scale factor to make wavefunctions visible on the energy axis

    # ==========================================
    # PLOT 1: The Physical Spectrum (S, P, D states)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(r, v_total, color="black", lw=2, label="Central Potential $V(r)$")

    # Plot 1S
    ax1.axhline(e1s_fd, color="blue", linestyle="--", alpha=0.4)
    ax1.fill_between(
        r, e1s_fd, e1s_fd + scale * u1s_fd**2, color="blue", alpha=0.3, label="1S State"
    )

    # Plot 1P
    ax1.axhline(e1p_fd_bare, color="green", linestyle="--", alpha=0.4)
    ax1.fill_between(
        r,
        e1p_fd_bare,
        e1p_fd_bare + scale * u1p_fd**2,
        color="green",
        alpha=0.3,
        label="1P State",
    )

    # Plot 2S
    ax1.axhline(e2s_fd, color="red", linestyle="--", alpha=0.4)
    ax1.fill_between(
        r, e2s_fd, e2s_fd + scale * u2s_fd**2, color="red", alpha=0.3, label="2S State"
    )

    # Plot 1D
    ax1.axhline(e1d_fd_bare, color="purple", linestyle="--", alpha=0.4)
    ax1.fill_between(
        r,
        e1d_fd_bare,
        e1d_fd_bare + scale * u1d_fd**2,
        color="purple",
        alpha=0.3,
        label="1D State",
    )

    display_limit = 7.0 / max(0.5, B)
    ax1.set_xlim(0, display_limit)
    ax1.set_ylim(-3.0, e1d_fd_bare + 1.0)
    ax1.set_title(
        f"Bottomonium Radial Spectrum and Wavefunction Densities\n($\\alpha_s$={ALPHA_S}, b={B}, c={C})",
        fontsize=14,
    )
    ax1.set_xlabel("Radius $r$ (fm/GeV$^{-1}$)", fontsize=12)
    ax1.set_ylabel("Energy (GeV)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=11)
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(f"{fig_dir}/spectrum_physics.png", dpi=300)

    # ==========================================
    # PLOT 2: GEM Validation (GEM vs Exact FD)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # 1S Comparison
    ax2.plot(r, u1s_fd**2, color="blue", lw=3, alpha=0.6, label="Exact Numerical 1S")
    ax2.plot(
        r,
        u1s_gem**2,
        color="black",
        linestyle="--",
        lw=1.5,
        label="GEM Approximation 1S",
    )

    # 2S Comparison
    ax2.plot(r, u2s_fd**2, color="red", lw=3, alpha=0.6, label="Exact Numerical 2S")
    ax2.plot(
        r, u2s_gem**2, color="black", linestyle=":", lw=2, label="GEM Approximation 2S"
    )

    ax2.set_xlim(0, 5.0 / max(0.5, B))
    ax2.set_title(
        "Validation: Gaussian Expansion Method vs Exact Numerical FD", fontsize=14
    )
    ax2.set_xlabel("Radius $r$", fontsize=12)
    ax2.set_ylabel("Probability Density $|u(r)|^2$", fontsize=12)
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(f"{fig_dir}/gem_validation.png", dpi=300)

    # ==========================================
    # PLOT 3: Mass Percent Error Bar Chart
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    x_pos_all = np.arange(len(results["methods_all"]))
    width = 0.35

    ax3.bar(
        x_pos_all - width / 2,
        results["mass_pct_errors_1s"],
        width,
        label="1S Mass % Error",
        color="darkblue",
        alpha=0.7,
    )
    ax3.bar(
        x_pos_all + width / 2,
        results["mass_pct_errors_2s"],
        width,
        label="2S Mass % Error",
        color="darkred",
        alpha=0.7,
    )

    ax3.set_yscale("log")
    ax3.set_xticks(x_pos_all)
    ax3.set_xticklabels(results["methods_all"], rotation=25, ha="right", fontsize=10)
    ax3.set_title(
        "Phenomenological Mass Error vs Experimental Data (Log Scale)", fontsize=14
    )
    ax3.set_ylabel("Percent Error (%)", fontsize=12)
    ax3.legend()
    ax3.grid(True, axis="y", alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(f"{fig_dir}/variational_errors.png", dpi=300)
    print(f"\nPlots successfully generated and saved in '{fig_dir}/'")

    # ==========================================
    # EXPORT RESULTS TO CSV TABLE
    # ==========================================
    os.makedirs("results/tables", exist_ok=True)
    csv_path = "results/tables/cornell_3d_toy_numerical_results.csv"

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Method",
                "E_1S (GeV)",
                "E_2S (GeV)",
                "Mass 1S (GeV)",
                "Mass 1S Diff (MeV)",
                "Mass 1S % Error",
                "Mass 2S (GeV)",
                "Mass 2S Diff (MeV)",
                "Mass 2S % Error",
            ]
        )

        for i, method in enumerate(results["methods_all"]):
            writer.writerow(
                [
                    method,
                    f"{results['energies_1s'][i]:.5f}",
                    f"{results['energies_2s'][i]:.5f}",
                    f"{results['masses_1s'][i]:.5f}",
                    f"{results['mass_diffs_1s'][i]:.2f}",
                    f"{results['mass_pct_errors_1s'][i]:.3f}%",
                    f"{results['masses_2s'][i]:.5f}",
                    f"{results['mass_diffs_2s'][i]:.2f}",
                    f"{results['mass_pct_errors_2s'][i]:.3f}%",
                ]
            )

    print(f"Numerical results table successfully generated and saved as '{csv_path}'")
    plt.show()


if __name__ == "__main__":
    create_showcase()
