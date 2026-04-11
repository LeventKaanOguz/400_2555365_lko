import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.pipeline import run_cornell_pipeline

# --- System Parameters ---
HBAR = 1.0
M_Q = 1.50  # Charm quark mass in GeV (Standard phenomenological value)
MU = M_Q / 2.0
ALPHA_S = 0.40  # Strong coupling constant for the Charmonium interaction
B = 0.183  # Linear confinement string tension
C = -0.250  # Constant potential shift (Adjusted for Charmonium)
SIGMA_SMEAR = 1.5  # Smearing parameter for spin-spin interaction (GeV)
EXP_MASS_1S = 2.9839  # eta_c (1S) mass in GeV
EXP_MASS_1S_TRIPLET = 3.0969  # J/psi (1S) mass in GeV
EXP_MASS_2S = 3.6861  # psi(2S) mass in GeV


def run_comparisons():
    results = run_cornell_pipeline(
        m_q=M_Q,
        alpha_s=ALPHA_S,
        b=B,
        c=C,
        sigma_smear=SIGMA_SMEAR,
        exp_mass_1s=EXP_MASS_1S,
        exp_mass_1s_triplet=EXP_MASS_1S_TRIPLET,
        exp_mass_2s=EXP_MASS_2S,
        hbar=HBAR,
        system_name="Charmonium",
    )

    evals_gem = results["evals_gem"]
    u_gem = results["u_gem"]
    evals_gem_1 = results["evals_gem_1"]
    u_gem_1 = results["u_gem_1"]
    evals_gem_2 = results["evals_gem_2"]
    u_gem_2 = results["u_gem_2"]
    calc_hf_shift = results["calc_hf_shift"]
    calc_so_shift = results["calc_so_shift"]

    def get_mass(evals, u_arr, state_idx, spin, l, j=None):
        bare = evals[state_idx]
        # Hyperfine splitting is zero for L > 0 since wavefunctions vanish at the origin
        hf_shift = calc_hf_shift(u_arr[:, state_idx], spin=spin) if l == 0 else 0.0

        so_shift = 0.0
        if l > 0 and spin > 0 and j is not None:
            so_shift = calc_so_shift(u_arr[:, state_idx], l=l, s=spin, j=j)

        return 2 * M_Q + bare + hf_shift + so_shift

    table_data = [
        (
            "(1^1S) η_c",
            get_mass(evals_gem, u_gem, 0, 0, 0),
            "0.4860",
            "3.0330",
            "2.9839±0.0004",
            "2.9644",
            "2.981",
            "3.068",
            "2.980",
        ),
        (
            "(1^3S) J/ψ",
            get_mass(evals_gem, u_gem, 0, 1, 0),
            "0.4207",
            "3.117",
            "3.0969±0.000006",
            "3.0964",
            "3.096",
            "-",
            "3.096",
        ),
        (
            "(1^1P) h_c",
            get_mass(evals_gem_1, u_gem_1, 0, 0, 1),
            "0.3296",
            "3.5260",
            "3.52538±0.00011",
            "3.4161",
            "3.525",
            "3.534",
            "-",
        ),
        (
            "(1^3P_0) χ_c0",
            get_mass(evals_gem_1, u_gem_1, 0, 1, 1, j=0),
            "0.3246",
            "3.5319",
            "-",
            "3.4358",
            "3.555",
            "-",
            "-",
        ),
        (
            "(1^3P_1) χ_c1",
            get_mass(evals_gem_1, u_gem_1, 0, 1, 1, j=1),
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ),
        (
            "(1^3P_2) χ_c2",
            get_mass(evals_gem_1, u_gem_1, 0, 1, 1, j=2),
            "-",
            "-",
            "3.55617±0.00007",
            "-",
            "-",
            "-",
            "-",
        ),
        (
            "(1^1D) η_c2",
            get_mass(evals_gem_2, u_gem_2, 0, 0, 2),
            "0.2919",
            "3.8040",
            "-",
            "3.6751",
            "3.807",
            "3.802",
            "-",
        ),
        (
            "(1^3D) ψ",
            get_mass(evals_gem_2, u_gem_2, 0, 1, 2, j=1),
            "0.2915",
            "3.8044",
            "3.77313±0.0004",
            "3.6881",
            "3.783",
            "-",
            "-",
        ),
        (
            "(2^1S) η_c",
            get_mass(evals_gem, u_gem, 1, 0, 0),
            "0.40908",
            "3.6236",
            "3.6375±0.0011",
            "3.5078",
            "3.635",
            "3.638",
            "3.624",
        ),
        (
            "(2^3S) ψ(2S)",
            get_mass(evals_gem, u_gem, 1, 1, 0),
            "0.3940",
            "3.6678",
            "3.68610±0.00006",
            "3.605",
            "3.685",
            "-",
            "3.727",
        ),
        (
            "(2^1P) h_c",
            get_mass(evals_gem_1, u_gem_1, 1, 0, 1),
            "0.2610",
            "3.9335",
            "-",
            "3.8774",
            "3.926",
            "3.936",
            "-",
        ),
        (
            "(2^3P_0) χ_c0",
            get_mass(evals_gem_1, u_gem_1, 1, 1, 1, j=0),
            "0.2578",
            "3.9413",
            "-",
            "3.9011",
            "3.949",
            "-",
            "-",
        ),
        (
            "(2^3P_1) χ_c1",
            get_mass(evals_gem_1, u_gem_1, 1, 1, 1, j=1),
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ),
        (
            "(2^3P_2) χ_c2",
            get_mass(evals_gem_1, u_gem_1, 1, 1, 1, j=2),
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ),
        (
            "(2^1D) η_c2",
            get_mass(evals_gem_2, u_gem_2, 1, 0, 2),
            "0.2024",
            "4.1573",
            "-",
            "-",
            "4.196",
            "4.150",
            "-",
        ),
        (
            "(2^3D) ψ",
            get_mass(evals_gem_2, u_gem_2, 1, 1, 2, j=1),
            "0.2020",
            "4.1582",
            "-",
            "-",
            "4.150",
            "-",
            "-",
        ),
    ]

    output_lines = []
    output_lines.append("--- Comparison with Literature (Akbar et al. 2024) ---")
    output_lines.append(
        "Reference paper provided Experimental and Theoretical benchmarks for charmonium."
    )
    output_lines.append(
        f"{'State':<15} | {'Our Work (GEM)':<15} | {'Akbar Var Param':<15} | {'Akbar (2024)':<12} | {'Exp [27]':<18} | {'[33]':<8} | {'[34]':<8} | {'[35]':<8} | {'[36]':<8}"
    )
    output_lines.append("-" * 125)

    for row in table_data:
        output_lines.append(
            f"{row[0]:<15} | {row[1]:<15.4f} | {row[2]:<15} | {row[3]:<12} | {row[4]:<18} | {row[5]:<8} | {row[6]:<8} | {row[7]:<8} | {row[8]:<8}"
        )
    output_lines.append("=" * 125 + "\n")

    output_lines.append("--- Error Analysis vs Experimental Data ---")
    output_lines.append(
        f"{'State':<15} | {'Calculated [GeV]':<18} | {'Experimental [GeV]':<18} | {'Abs Error [MeV]':<15} | {'% Error':<10}"
    )
    output_lines.append("-" * 88)

    for row in table_data:
        state = row[0]
        calc_mass = row[1]
        exp_str = row[4]

        if exp_str != "-":
            exp_mass = float(exp_str.split("±")[0])
            abs_err = (calc_mass - exp_mass) * 1000.0  # Convert to MeV
            pct_err = abs(calc_mass - exp_mass) / exp_mass * 100.0
            output_lines.append(
                f"{state:<15} | {calc_mass:<18.4f} | {exp_mass:<18.4f} | {abs_err:<15.1f} | {pct_err:<10.3f}"
            )
        else:
            output_lines.append(
                f"{state:<15} | {calc_mass:<18.4f} | {'-':<18} | {'-':<15} | {'-':<10}"
            )
    output_lines.append("=" * 88 + "\n")

    output_text = "\n".join(output_lines)
    print(output_text)

    os.makedirs("results/tables", exist_ok=True)
    out_path = "results/tables/charmonium_error_analysis.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"Error analysis table successfully saved to '{out_path}'\n")

    results["comparison_table_data"] = table_data
    return results


if __name__ == "__main__":
    run_comparisons()
