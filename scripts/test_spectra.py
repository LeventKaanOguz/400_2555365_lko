#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from quarkonia.pdg_loader import load_pdg_data
from quarkonia import paths


def evaluate_decay_error(calculated_width, state_label, sector, decay_type):
    """
    Automatically compares your GEM decay width to the PDG data.

    sector: 'cc' or 'bb'
    decay_type: 'ee_keV', 'gammagamma_keV', or 'total_MeV'
    """
    pdg = load_pdg_data()

    db_key = f"{sector}_widths_{decay_type}"

    if db_key in pdg and state_label in pdg[db_key]:
        exp_width = pdg[db_key][state_label]

        if exp_width is not None and exp_width > 0:
            pct_error = abs(calculated_width - exp_width) / exp_width * 100.0
            print(
                f"  [{state_label}] {decay_type} Error: {pct_error:.2f}% (Exp: {exp_width}, Calc: {calculated_width:.3f})"
            )
            return pct_error

    return None


def run_benchmarks():
    sectors = [
        ("bb", "Bottomonium (b_bbar)"),
        ("cc", "Charmonium (c_cbar)"),
        ("bc", "B_c Meson (b_cbar)"),
    ]

    print("--- Theoretical vs Experimental Benchmarks (MSE) ---")
    for sect_prefix, sector in sectors:
        csv_path = paths.errors_csv(sect_prefix)
        if not os.path.exists(csv_path):
            print(f"[{sector}] Error file not found. Run run_spectrum.py first.")
            continue

        df = pd.read_csv(csv_path)
        df_valid = df[df["Exp_GeV"] != "N/A"].copy()
        if df_valid.empty:
            print(f"[{sector}] No valid experimental data points to compare.")
            continue

        mse = (
            np.mean(
                (
                    df_valid["Calculated_GeV"].astype(float)
                    - df_valid["Exp_GeV"].astype(float)
                )
                ** 2
            )
            * 1e6
        )
        rmse = np.sqrt(mse)

        print(
            f"{sector}:\n  -> Mass MSE  = {mse:.2f} MeV^2\n  -> Mass RMSE = {rmse:.2f} MeV"
        )

        # Evaluate observables
        obs_csv_path = paths.observables_csv(sect_prefix)
        if os.path.exists(obs_csv_path):
            df_obs = pd.read_csv(obs_csv_path)

            for _, row in df_obs.iterrows():
                state_full = row["State"]
                state_label = state_full.split()[
                    0
                ]  # Extracts "(1^3S)" from "(1^3S) J/\psi"
                obs_type = row["Observable_Type"]
                calc_val = float(row["Value_keV"])

                if "e+e-" in obs_type:
                    evaluate_decay_error(calc_val, state_label, sect_prefix, "ee_keV")
                elif "γγ" in obs_type or "gamma" in obs_type:
                    evaluate_decay_error(
                        calc_val, state_label, sect_prefix, "gammagamma_keV"
                    )
        print()


if __name__ == "__main__":
    run_benchmarks()
