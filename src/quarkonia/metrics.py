import csv
import os
import pandas as pd
import numpy as np


def format_and_evaluate(calculated_states, pdg_data, sector_name):
    print(f"\n--- Error Analysis vs Experimental Data ({sector_name}) ---")
    print(
        f"{'State':<18} | {'Calculated [GeV]':<18} | {'Experimental [GeV]':<18} | {'Abs Error [MeV]':<15} | {'% Error':<10}"
    )
    print("-" * 88)

    results = []
    for state, calc_mass in calculated_states.items():
        pure_state = state.split()[0]
        exp_mass = pdg_data.get(pure_state, None)

        if exp_mass is not None:
            abs_err = (calc_mass - exp_mass) * 1000.0
            pct_err = abs(calc_mass - exp_mass) / exp_mass * 100.0
            print(
                f"{state:<18} | {calc_mass:<18.4f} | {exp_mass:<18.4f} | {abs_err:<15.1f} | {pct_err:<10.3f}"
            )
            results.append([state, calc_mass, exp_mass, abs_err, pct_err])
        else:
            print(
                f"{state:<18} | {calc_mass:<18.4f} | {'-':<18} | {'-':<15} | {'-':<10}"
            )
            results.append([state, calc_mass, "N/A", "N/A", "N/A"])

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "results")
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{sector_name}_errors.csv"), "w", newline="") as f:
        csv.writer(f).writerows(
            [["State", "Calculated_GeV", "Exp_GeV", "Abs_Err_MeV", "Pct_Err"]] + results
        )


def export_gem_parameters(nu_array, evecs, l_str, sector_name):
    """Exports the compact 25-row analytical GEM coefficients."""
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "results")
    )
    os.makedirs(out_dir, exist_ok=True)

    data_gem = {
        "Index": np.arange(len(nu_array)),
        "nu_width": nu_array,
        f"c_1{l_str}": evecs[:, 0],
        f"c_2{l_str}": evecs[:, 1] if evecs.shape[1] > 1 else np.zeros_like(nu_array),
        f"c_3{l_str}": evecs[:, 2] if evecs.shape[1] > 2 else np.zeros_like(nu_array),
    }

    df_gem = pd.DataFrame(data_gem)
    df_gem.to_csv(
        os.path.join(out_dir, f"{sector_name}_{l_str}_Wave_GEM_Coefficients.csv"),
        index=False,
    )
