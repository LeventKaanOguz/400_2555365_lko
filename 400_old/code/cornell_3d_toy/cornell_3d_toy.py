import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.pipeline import run_cornell_pipeline

# --- System Parameters ---
HBAR = 1.0
M_Q = 4.730
MU = M_Q / 2.0
ALPHA_S = 0.3807  # Strong coupling constant for the Coulomb-like interaction
B = 0.183  # Linear confinement string tension
C = 0.070  # Constant potential shift
SIGMA_SMEAR = 1.5  # Smearing parameter for spin-spin interaction (GeV)
EXP_MASS_1S = 9.3987  # eta_b (1S) mass in GeV
EXP_MASS_1S_TRIPLET = 9.4603  # Upsilon (1S) mass in GeV
EXP_MASS_2S = 10.0234  # Upsilon (2S) mass in GeV


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
        system_name="Bottomonium",
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
            "(1^1S) η_b",
            get_mass(evals_gem, u_gem, 0, 0, 0),
            "0.7828",
            "9.5535",
            "9.3987±0.002",
            "9.5615",
            "9.398",
            "9.398",
            "9.5079",
            "9.452",
            "-",
        ),
        (
            "(1^3S) Υ_b",
            get_mass(evals_gem, u_gem, 0, 1, 0),
            "0.7571",
            "9.5722",
            "9.4603±0.00026",
            "9.6478",
            "9.478",
            "9.460",
            "9.5229",
            "9.480",
            "-",
        ),
        (
            "(1^1P) h_b",
            get_mass(evals_gem_1, u_gem_1, 0, 0, 1),
            "0.5129",
            "9.9373",
            "9.8993",
            "9.9324",
            "9.900",
            "9.894",
            "9.9279",
            "-",
            "-",
        ),
        (
            "(1^3P_0) χ_b0",
            get_mass(evals_gem_1, u_gem_1, 0, 1, 1, j=0),
            "0.5096",
            "9.9391",
            "9.8594",
            "9.9389",
            "9.912",
            "9.858",
            "9.9232",
            "-",
            "-",
        ),
        (
            "(1^3P_1) χ_b1",
            get_mass(evals_gem_1, u_gem_1, 0, 1, 1, j=1),
            "-",
            "-",
            "9.8928",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ),
        (
            "(1^3P_2) χ_b2",
            get_mass(evals_gem_1, u_gem_1, 0, 1, 1, j=2),
            "-",
            "-",
            "9.9122",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ),
        (
            "(1^1D) η_b2",
            get_mass(evals_gem_2, u_gem_2, 0, 0, 2),
            "0.4425",
            "10.1398",
            "-",
            "-",
            "10.163",
            "-",
            "10.1355",
            "-",
            "-",
        ),
        (
            "(1^3D) Υ",
            get_mass(evals_gem_2, u_gem_2, 0, 1, 2, j=2),
            "0.4422",
            "10.1399",
            "-",
            "-",
            "10.161",
            "-",
            "10.1548",
            "-",
            "-",
        ),
        (
            "(2^1S) η_b",
            get_mass(evals_gem, u_gem, 1, 0, 0),
            "0.62615",
            "9.9980",
            "-",
            "-",
            "9.990",
            "10.017",
            "10.0041",
            "10.030",
            "-",
        ),
        (
            "(2^3S) Υ",
            get_mass(evals_gem, u_gem, 1, 1, 0),
            "0.6215",
            "10.0052",
            "10.0233±0.0003",
            "10.0167",
            "10.023",
            "10.356",
            "10.0101",
            "10.055",
            "-",
        ),
        (
            "(2^1P) h_b",
            get_mass(evals_gem_1, u_gem_1, 1, 0, 1),
            "0.3924",
            "10.2210",
            "10.2598",
            "10.2161",
            "10.260",
            "10.259",
            "-",
            "-",
            "-",
        ),
        (
            "(2^3P_0) χ_b0",
            get_mass(evals_gem_1, u_gem_1, 1, 1, 1, j=0),
            "0.3909",
            "10.2288",
            "10.2325",
            "-",
            "10.2232",
            "10.255",
            "-",
            "-",
            "-",
        ),
        (
            "(2^3P_1) χ_b1",
            get_mass(evals_gem_1, u_gem_1, 1, 1, 1, j=1),
            "-",
            "-",
            "10.2555",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ),
        (
            "(2^3P_2) χ_b2",
            get_mass(evals_gem_1, u_gem_1, 1, 1, 1, j=2),
            "-",
            "-",
            "10.2687",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ),
        (
            "(2^2D) η_b2",
            get_mass(evals_gem_2, u_gem_2, 1, 0, 2),
            "0.301959",
            "10.3780",
            "-",
            "-",
            "-",
            "10.450",
            "-",
            "-",
            "-",
        ),
        (
            "(2^3D) Υ",
            get_mass(evals_gem_2, u_gem_2, 1, 1, 2, j=2),
            "0.30170",
            "10.3783",
            "-",
            "-",
            "10.443",
            "10.442",
            "-",
            "-",
            "-",
        ),
    ]

    output_lines = []
    output_lines.append("--- Comparison with Literature (Akbar et al. 2024) ---")
    output_lines.append(
        "Reference paper provided Experimental and Theoretical benchmarks for bottomonium."
    )
    output_lines.append(
        f"{'State':<15} | {'Our Work (GEM)':<15} | {'Akbar Var Param':<15} | {'Akbar (2024)':<12} | {'Experimental':<18} | {'[27]':<8} | {'[33]':<8} | {'[34]':<8} | {'[35]':<8} | {'[25]':<8} | {'[36]':<4}"
    )
    output_lines.append("-" * 147)

    for row in table_data:
        output_lines.append(
            f"{row[0]:<15} | {row[1]:<15.4f} | {row[2]:<15} | {row[3]:<12} | {row[4]:<18} | {row[5]:<8} | {row[6]:<8} | {row[7]:<8} | {row[8]:<8} | {row[9]:<8} | {row[10]:<4}"
        )
    output_lines.append("=" * 147 + "\n")

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
    out_path = "results/tables/cornell_3d_toy_error_analysis.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"Error analysis table successfully saved to '{out_path}'\n")

    results["comparison_table_data"] = table_data
    return results


if __name__ == "__main__":
    run_comparisons()
