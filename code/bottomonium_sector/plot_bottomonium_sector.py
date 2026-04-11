import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from bottomonium_sector import run_comparisons, ALPHA_S, B, C


def create_showcase():
    print("Gathering data from bottomonium_sector.py...")
    results = run_comparisons()

    r = results["r"]
    v_total = results["v_total"]

    # S-Wave States
    e1s_gem_bare = results["evals_gem"][0]
    e2s_gem_bare = results["evals_gem"][1]
    u1s_gem = results["u1s_gem"]
    u2s_gem = results["u2s_gem"]

    # Orbital Excitations (P and D waves)
    e1p_gem_bare = results["evals_gem_1"][0]
    u1p_gem = results["u_gem_1"][:, 0]
    e1d_gem_bare = results["evals_gem_2"][0]
    u1d_gem = results["u_gem_2"][:, 0]

    # --- Setup Output Directory ---
    fig_dir = "results/figures/bottomonium_sector"
    os.makedirs(fig_dir, exist_ok=True)

    scale = 0.6  # Scale factor to make wavefunctions visible on the energy axis

    # ==========================================
    # PLOT 1: The Physical Spectrum (S, P, D states)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(r, v_total, color="black", lw=2, label="Central Potential $V(r)$")

    # Plot 1S
    ax1.axhline(e1s_gem_bare, color="blue", linestyle="--", alpha=0.4)
    ax1.fill_between(
        r,
        e1s_gem_bare,
        e1s_gem_bare + scale * u1s_gem**2,
        color="blue",
        alpha=0.3,
        label="1S State (GEM)",
    )

    # Plot 1P
    ax1.axhline(e1p_gem_bare, color="green", linestyle="--", alpha=0.4)
    ax1.fill_between(
        r,
        e1p_gem_bare,
        e1p_gem_bare + scale * u1p_gem**2,
        color="green",
        alpha=0.3,
        label="1P State (GEM)",
    )

    # Plot 2S
    ax1.axhline(e2s_gem_bare, color="red", linestyle="--", alpha=0.4)
    ax1.fill_between(
        r,
        e2s_gem_bare,
        e2s_gem_bare + scale * u2s_gem**2,
        color="red",
        alpha=0.3,
        label="2S State (GEM)",
    )

    # Plot 1D
    ax1.axhline(e1d_gem_bare, color="purple", linestyle="--", alpha=0.4)
    ax1.fill_between(
        r,
        e1d_gem_bare,
        e1d_gem_bare + scale * u1d_gem**2,
        color="purple",
        alpha=0.3,
        label="1D State (GEM)",
    )

    display_limit = 7.0 / max(0.5, B)
    ax1.set_xlim(0, display_limit)
    ax1.set_ylim(-3.0, e1d_gem_bare + 1.0)
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
    # PLOT 2: Mass Percent Error Bar Chart
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    x_pos_all = np.arange(len(results["methods_all"]))
    width = 0.35

    ax2.bar(
        x_pos_all - width / 2,
        results["mass_pct_errors_1s"],
        width,
        label="1S Mass % Error",
        color="darkblue",
        alpha=0.7,
    )
    ax2.bar(
        x_pos_all + width / 2,
        results["mass_pct_errors_2s"],
        width,
        label="2S Mass % Error",
        color="darkred",
        alpha=0.7,
    )

    ax2.set_yscale("log")
    ax2.set_xticks(x_pos_all)
    ax2.set_xticklabels(results["methods_all"], rotation=25, ha="right", fontsize=10)
    ax2.set_title(
        "Phenomenological Mass Error vs Experimental Data (Log Scale)", fontsize=14
    )
    ax2.set_ylabel("Percent Error (%)", fontsize=12)
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(f"{fig_dir}/variational_errors.png", dpi=300)

    # ==========================================
    # PLOT 3: Literature Mass Comparison
    # ==========================================
    if "comparison_table_data" in results:
        fig3, ax3 = plt.subplots(figsize=(12, 8))

        table_data = results["comparison_table_data"]
        states, our_masses, akbar_masses, exp_masses, exp_errors = [], [], [], [], []

        for row in table_data:
            exp_str = row[4]
            if exp_str != "-":
                states.append(row[0])
                our_masses.append(row[1])
                akbar_masses.append(float(row[3]) if row[3] != "-" else np.nan)

                parts = exp_str.split("±")
                exp_masses.append(float(parts[0]))
                exp_errors.append(float(parts[1]) if len(parts) > 1 else 0)

        x_pos = np.arange(len(states))
        width = 0.25

        ax3.bar(
            x_pos - width,
            our_masses,
            width,
            label="Our Work (GEM)",
            color="cornflowerblue",
        )
        ax3.bar(
            x_pos, akbar_masses, width, label="Akbar (2024) [Ref]", color="seagreen"
        )
        # Plot horizontal lines for experimental values for easier comparison
        ax3.hlines(
            exp_masses,
            x_pos - 1.5 * width,
            x_pos + 1.5 * width,
            colors="black",
            linestyles="solid",
            lw=1,
            label="Experimental (PDG)",
        )
        # Overlay the error bars on the line
        ax3.errorbar(
            x_pos + width,
            exp_masses,
            yerr=exp_errors,
            fmt="none",
            color="black",
            capsize=4,
        )

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(states, rotation=30, ha="right")
        ax3.set_ylabel("Mass (GeV)")
        ax3.set_title("Comparison of Calculated and Experimental Meson Masses")
        ax3.legend()
        ax3.grid(True, axis="y", linestyle="--", alpha=0.6)

        fig3.tight_layout()
        fig3.savefig(f"{fig_dir}/mass_comparison_literature.png", dpi=300)

    # ==========================================
    # PLOT 4: GEM Expansion Basis Functions (All Bases for 1S & 2S)
    # ==========================================
    if "nu_gem" in results:
        fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(18, 8))
        from scipy.integrate import simpson

        nu_gem = results["nu_gem"]
        evecs_gem = results["evecs_gem"]
        n_basis = results["n_basis_gem"]

        basis = []
        for n in nu_gem:
            # Reconstruct the normalized basis functions for l=0 (1S & 2S)
            u = r * np.exp(-n * r**2)
            norm = np.sqrt(simpson(y=u**2, x=r))
            basis.append(u / norm)

        # 1S State Plot
        ax4_1.plot(r, u1s_gem, color="black", lw=3, label="Total 1S Wavefunction (GEM)")

        components_1s = [(i, evecs_gem[i, 0] * basis[i]) for i in range(n_basis)]
        # Sort components solely to order the legend by prominence
        components_1s.sort(key=lambda x: np.max(np.abs(x[1])), reverse=True)

        for i, comp in components_1s:
            ax4_1.plot(
                r,
                comp,
                linestyle="--",
                alpha=0.7,
                label=f"Basis {i:02d} (c={evecs_gem[i, 0]:.3f}, $\\nu$={nu_gem[i]:.1e})",
            )

        ax4_1.set_xlim(0, 4.0 / max(0.5, B))
        ax4_1.set_title("Gaussian Expansion Method - 1S Basis Components", fontsize=14)
        ax4_1.set_xlabel("Radius $r$ (fm/GeV$^{-1}$)", fontsize=12)
        ax4_1.set_ylabel("Amplitude $u(r)$", fontsize=12)
        ax4_1.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        ax4_1.grid(True, alpha=0.3)

        # 2S State Plot
        ax4_2.plot(r, u2s_gem, color="red", lw=3, label="Total 2S Wavefunction (GEM)")

        components_2s = [(i, evecs_gem[i, 1] * basis[i]) for i in range(n_basis)]
        # Sort components solely to order the legend by prominence
        components_2s.sort(key=lambda x: np.max(np.abs(x[1])), reverse=True)

        for i, comp in components_2s:
            ax4_2.plot(
                r,
                comp,
                linestyle="--",
                alpha=0.7,
                label=f"Basis {i:02d} (c={evecs_gem[i, 1]:.3f}, $\\nu$={nu_gem[i]:.1e})",
            )

        ax4_2.set_xlim(0, 6.0 / max(0.5, B))
        ax4_2.set_title("Gaussian Expansion Method - 2S Basis Components", fontsize=14)
        ax4_2.set_xlabel("Radius $r$ (fm/GeV$^{-1}$)", fontsize=12)
        ax4_2.set_ylabel("Amplitude $u(r)$", fontsize=12)
        ax4_2.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        ax4_2.grid(True, alpha=0.3)

        fig4.tight_layout()
        fig4.savefig(f"{fig_dir}/gem_expansion_1s_2s.png", dpi=300)

    print(f"\nPlots successfully generated and saved in '{fig_dir}/'")

    # ==========================================
    # EXPORT RESULTS TO CSV TABLE
    # ==========================================
    os.makedirs("results/tables", exist_ok=True)
    csv_path = "results/tables/bottomonium_sector_numerical_results.csv"

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
