#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import json


def evaluate_decay_error(calculated_width, state_label, sector, decay_type):
    """
    Automatically compares your GEM decay width to the PDG data.

    sector: 'cc' or 'bb'
    decay_type: 'ee_keV', 'gammagamma_keV', or 'total_MeV'
    """
    json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "pdg_data.json")
    )
    with open(json_path, "r") as f:
        pdg = json.load(f)

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
    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )

    sectors = ["Bottomonium (b_bbar)", "Charmonium (c_cbar)", "B_c Meson (b_cbar)"]

    print("--- Theoretical vs Experimental Benchmarks (MSE) ---")
    for sector in sectors:
        csv_path = os.path.join(results_dir, f"{sector}_errors.csv")
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
        obs_csv_path = os.path.join(results_dir, f"{sector}_observables.csv")
        if os.path.exists(obs_csv_path):
            df_obs = pd.read_csv(obs_csv_path)
            sect_prefix = (
                "bb" if "b_bbar" in sector else ("cc" if "c_cbar" in sector else "bc")
            )

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
