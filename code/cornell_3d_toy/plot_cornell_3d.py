import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from scipy.integrate import simpson
from cornell_3d_toy import run_comparisons, ALPHA_S, B, C


def create_showcase():
    print("Gathering data from cornell_3d_toy.py...")
    results = run_comparisons()

    r = results["r"]
    v_total = results["v_total"]
    e1s_fd = results["e1s_fd"]
    e2s_fd = results["e2s_fd"]
    u1s_fd = results["u1s_fd"]
    u2s_fd = results["u2s_fd"]
    u1s_var = results["u1s_var"]
    u2s_var = results["u2s_var"]
    u1s_var_hyd = results["u1s_var_hyd"]
    u2s_var_hyd = results["u2s_var_hyd"]
    u1s_var_pow = results["u1s_var_pow"]
    u2s_var_pow = results["u2s_var_pow"]
    u1s_gem = results["u1s_gem"]
    u2s_gem = results["u2s_gem"]

    # --- Setup Output Directory ---
    fig_dir = "results/figures/cornell_3d_toy"

    # ==========================================
    # PLOT 1: Potential and Wavefunctions
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(r, v_total, color="black", lw=2, label="V(r) = -(4/3)α_s/r + br + c")

    ax1.axhline(
        e1s_fd, color="blue", linestyle="--", alpha=0.5, label=f"E_1S = {e1s_fd:.3f}"
    )
    ax1.axhline(
        e2s_fd, color="red", linestyle="--", alpha=0.5, label=f"E_2S = {e2s_fd:.3f}"
    )

    scale = 0.5
    ax1.fill_between(
        r,
        e1s_fd,
        e1s_fd + scale * u1s_fd**2,
        color="blue",
        alpha=0.3,
        label="|u_1S FD|²",
    )
    ax1.plot(
        r,
        e1s_fd + scale * u1s_var**2,
        color="cyan",
        linestyle=":",
        lw=2.5,
        label="|u_1S Var (Harmonic)|²",
    )
    ax1.plot(
        r,
        e1s_fd + scale * u1s_var_hyd**2,
        color="green",
        linestyle="-.",
        lw=2.0,
        label="|u_1S Var (Hydrogenic)|²",
    )
    ax1.plot(
        r,
        e1s_fd + scale * u1s_var_pow**2,
        color="olive",
        linestyle="-.",
        lw=2.0,
        label="|u_1S Var (Power)|²",
    )
    ax1.plot(
        r,
        e1s_fd + scale * u1s_gem**2,
        color="purple",
        linestyle="--",
        lw=2.0,
        label="|u_1S Var (GEM)|²",
    )

    ax1.fill_between(
        r,
        e2s_fd,
        e2s_fd + scale * u2s_fd**2,
        color="red",
        alpha=0.3,
        label="|u_2S FD|²",
    )
    ax1.plot(
        r,
        e2s_fd + scale * u2s_var**2,
        color="orange",
        linestyle=":",
        lw=2.5,
        label="|u_2S Var (Harmonic)|²",
    )
    ax1.plot(
        r,
        e2s_fd + scale * u2s_var_hyd**2,
        color="magenta",
        linestyle="-.",
        lw=2.0,
        label="|u_2S Var (Hydrogenic)|²",
    )
    ax1.plot(
        r,
        e2s_fd + scale * u2s_var_pow**2,
        color="teal",
        linestyle="-.",
        lw=2.0,
        label="|u_2S Var (Power)|²",
    )
    ax1.plot(
        r,
        e2s_fd + scale * u2s_gem**2,
        color="brown",
        linestyle="--",
        lw=2.0,
        label="|u_2S Var (GEM)|²",
    )

    display_limit = 6.0 / max(0.5, B)
    ax1.set_xlim(0, display_limit)
    ax1.set_ylim(-3.0, e2s_fd + 1.5)  # Constrained well lower limit
    ax1.set_title(
        f"Reduced Radial Wavefunction Densities vs Potential\n(α_s={ALPHA_S}, b={B}, c={C})"
    )
    ax1.set_xlabel("Radius (r)")
    ax1.set_ylabel("Energy")
    ax1.legend(loc="upper right", fontsize="small")

    fig1.tight_layout()
    fig1.savefig(f"{fig_dir}/wavefunctions.png", dpi=300)

    # ==========================================
    # PLOT 2: Error Analysis Bar Chart
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(results["methods"]))
    width = 0.35

    ax2.bar(
        x_pos - width / 2,
        results["errors_1s"],
        width,
        label="E_1S Error",
        color="blue",
        alpha=0.7,
    )
    ax2.bar(
        x_pos + width / 2,
        results["errors_2s"],
        width,
        label="E_2S Error",
        color="red",
        alpha=0.7,
    )

    ax2.set_yscale("log")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results["methods"], rotation=15, ha="right")
    ax2.set_title("Absolute Energy Errors vs Numerical FD (Log Scale)")
    ax2.set_ylabel("Absolute Error (Log Scale)")
    ax2.legend()

    fig2.tight_layout()
    fig2.savefig(f"{fig_dir}/energy_errors.png", dpi=300)

    # ==========================================
    # PLOT 3: Mass Percent Error Bar Chart
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    x_pos_all = np.arange(len(results["methods_all"]))

    ax3.bar(
        x_pos_all - width / 2,
        results["mass_pct_errors_1s"],
        width,
        label="1S Mass % Error",
        color="purple",
        alpha=0.7,
    )
    ax3.bar(
        x_pos_all + width / 2,
        results["mass_pct_errors_2s"],
        width,
        label="2S Mass % Error",
        color="orange",
        alpha=0.7,
    )

    ax3.set_yscale("log")
    ax3.set_xticks(x_pos_all)
    ax3.set_xticklabels(results["methods_all"], rotation=15, ha="right")
    ax3.set_title("Mass Percent Error vs Experimental Data (Log Scale)")
    ax3.set_ylabel("Percent Error (%)")
    ax3.legend()

    fig3.tight_layout()
    fig3.savefig(f"{fig_dir}/mass_errors.png", dpi=300)
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
                "E_1S Absolute Error",
                "E_2S Absolute Error",
            ]
        )

        for i, method in enumerate(results["methods_all"]):
            e1s_val = results["energies_1s"][i]
            e2s_val = results["energies_2s"][i]
            m1s_val = results["masses_1s"][i]
            m2s_val = results["masses_2s"][i]
            d1s_val = results["mass_diffs_1s"][i]
            d2s_val = results["mass_diffs_2s"][i]
            p1s_val = results["mass_pct_errors_1s"][i]
            p2s_val = results["mass_pct_errors_2s"][i]

            str_err_1s = (
                "-"
                if method == "Numerical: Matrix FD"
                else f"{results['errors_1s'][i]:.2e}"
            )
            str_err_2s = (
                "-"
                if method == "Numerical: Matrix FD"
                else f"{results['errors_2s'][i]:.2e}"
            )

            writer.writerow(
                [
                    method,
                    f"{e1s_val:.5f}",
                    f"{e2s_val:.5f}",
                    f"{m1s_val:.5f}",
                    f"{d1s_val:.2f}",
                    f"{p1s_val:.3f}%",
                    f"{m2s_val:.5f}",
                    f"{d2s_val:.2f}",
                    f"{p2s_val:.3f}%",
                    str_err_1s,
                    str_err_2s,
                ]
            )

    print(f"Numerical results table successfully generated and saved as '{csv_path}'")
    plt.show()


if __name__ == "__main__":
    create_showcase()
