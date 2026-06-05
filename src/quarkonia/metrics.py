import csv
import os
import json
import pandas as pd
import numpy as np


def format_and_evaluate(calculated_states, pdg_data, sector_name, n_fit_params=4):
    """
    Tabulates calculated vs experimental masses and reports goodness-of-fit.

    In addition to the per-state absolute/percentage errors, this computes a
    chi-square statistic for the sector:

        chi^2 = sum_i [ (m_calc,i - m_exp,i) / sigma_i ]^2

    where sigma_i is the propagated mass uncertainty (from the fitted-parameter
    covariance, in MeV) combined in quadrature with a 1 MeV experimental/rounding
    floor so the statistic stays finite when no parameter uncertainty is
    available. The reduced chi^2 uses dof = N_states - n_fit_params. A converged
    fit with realistic uncertainties should give chi^2/dof of order 1.

    Parameters
    ----------
    calculated_states : dict
        Maps state name -> (mass_GeV, mass_err_GeV) or bare mass_GeV.
    pdg_data : dict
        Experimental PDG mass data keyed by the bare state label.
    sector_name : str
        Sector designation, used for the output CSV filename.
    n_fit_params : int, optional
        Number of free fit parameters, used for the degrees of freedom in the
        reduced chi-square, by default 4 (alpha_s, b, c, sigma).
    """
    print(f"\n--- Error Analysis vs Experimental Data ({sector_name}) ---")
    print(
        f"{'State':<18} | {'Calculated [GeV]':<25} | {'Experimental [GeV]':<18} | {'Abs Error [MeV]':<15} | {'% Error':<10} | {'Pull (sigma)':<12} | {'MSE [MeV^2]':<15}"
    )
    print("-" * 130)

    SIGMA_FLOOR_MEV = 1.0  # experimental/rounding floor to keep chi^2 finite

    results = []
    chi2 = 0.0
    sq_err_sum = 0.0
    n_compared = 0
    for state, calc_data in calculated_states.items():
        if isinstance(calc_data, tuple):
            calc_mass, mass_err = calc_data
        else:
            calc_mass, mass_err = calc_data, 0.0

        pure_state = state.split()[0]
        exp_mass = pdg_data.get(pure_state, None)

        calc_str = f"{calc_mass:.4f} ± {mass_err:.4f}"

        if exp_mass is not None:
            abs_err = (calc_mass - exp_mass) * 1000.0
            pct_err = abs(calc_mass - exp_mass) / exp_mass * 100.0
            mse = abs_err**2
            sigma_mev = np.hypot(mass_err * 1000.0, SIGMA_FLOOR_MEV)
            pull = abs_err / sigma_mev
            chi2 += pull**2
            sq_err_sum += mse
            n_compared += 1
            print(
                f"{state:<18} | {calc_str:<25} | {exp_mass:<18.4f} | {abs_err:<15.1f} | {pct_err:<10.3f} | {pull:<12.2f} | {mse:<15.2f}"
            )
            results.append(
                [state, calc_mass, mass_err, exp_mass, abs_err, pct_err, pull, mse]
            )
        else:
            print(
                f"{state:<18} | {calc_str:<25} | {'-':<18} | {'-':<15} | {'-':<10} | {'-':<12} | {'-':<15}"
            )
            results.append(
                [state, calc_mass, mass_err, "N/A", "N/A", "N/A", "N/A", "N/A"]
            )

    dof = max(n_compared - n_fit_params, 1)
    rmse = np.sqrt(sq_err_sum / n_compared) if n_compared else 0.0
    red_chi2 = chi2 / dof
    print("-" * 130)
    print(
        f"Goodness of fit ({sector_name}): N = {n_compared} states, "
        f"dof = {n_compared} - {n_fit_params} = {n_compared - n_fit_params}"
    )
    print(
        f"   chi^2 = {chi2:.2f}   chi^2/dof = {red_chi2:.3f}   "
        f"RMS mass deviation = {rmse:.2f} MeV"
    )

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "results")
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{sector_name}_errors.csv"), "w", newline="") as f:
        csv.writer(f).writerows(
            [
                [
                    "State",
                    "Calculated_GeV",
                    "Mass_Err_GeV",
                    "Exp_GeV",
                    "Abs_Err_MeV",
                    "Pct_Err",
                    "Pull_sigma",
                    "MSE_MeV2",
                ]
            ]
            + results
        )

    return {
        "sector": sector_name,
        "chi2": chi2,
        "dof": n_compared - n_fit_params,
        "chi2_per_dof": red_chi2,
        "rms_mev": rmse,
        "n": n_compared,
    }


def export_gem_parameters(nu_array, evecs, l_str, sector_name):
    """
    Exports the compact 25-row analytical GEM coefficients.

    Parameters
    ----------
    nu_array : numpy.ndarray
        Array of Gaussian basis widths.
    evecs : numpy.ndarray
        Matrix containing the computed eigenvectors.
    l_str : str
        String representation of the orbital angular momentum.
    sector_name : str
        Name of the corresponding particle sector.
    """
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


def export_observables(calculated_observables, sector_name):
    """
    Exports the computed decay observables to a CSV file.
    """
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "results")
    )
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{sector_name}_observables.csv")
    # Using utf-8 encoding to support unicode characters like γ
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["State", "Observable_Type", "Value_keV", "Error_keV"])

        for state, (obs_type, obs_val, obs_err) in calculated_observables.items():
            writer.writerow([state, obs_type, f"{obs_val:.4f}", f"{obs_err:.4f}"])


def generate_consolidated_report():
    """
    Reads the individual error and observable CSVs and cross-references them with pdg_data.json
    to generate a consolidated error report containing Abs Error, % Error, and MSE.
    """
    pdg_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "pdg_data.json")
    )
    with open(pdg_path, "r") as f:
        pdg_data = json.load(f)

    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "results")
    )

    sectors = [
        ("bb", "Bottomonium (b_bbar)"),
        ("cc", "Charmonium (c_cbar)"),
        ("bc", "B_c Meson (b_cbar)"),
        ("cu", "D Meson (c_ubar)"),
    ]

    report_data = []
    SIGMA_FLOOR = 1e-9  # avoid division by zero for the dimensionless pull

    # Process Masses
    for sector_id, sector_name in sectors:
        mass_csv = os.path.join(results_dir, f"{sector_name}_errors.csv")
        if os.path.exists(mass_csv):
            df_mass = pd.read_csv(mass_csv)
            df_valid = df_mass[df_mass["Exp_GeV"] != "N/A"].copy()
            for _, row in df_valid.iterrows():
                state = row["State"]
                calc_val = float(row["Calculated_GeV"])
                exp_val = float(row["Exp_GeV"])
                abs_err_mev = float(row["Abs_Err_MeV"])
                pct_err = float(row["Pct_Err"])
                mse = float(row["MSE_MeV2"])
                # Propagated 1-sigma uncertainty (MeV) and the pull (chi contribution)
                unc_mev = float(row.get("Mass_Err_GeV", 0.0)) * 1000.0
                pull = abs_err_mev / np.hypot(unc_mev, 1.0)  # 1 MeV exp/round floor

                report_data.append(
                    {
                        "Sector": sector_id.upper(),
                        "Property": "Mass (GeV)",
                        "State": state,
                        "Calculated": calc_val,
                        "Experimental": exp_val,
                        "Uncertainty": unc_mev,
                        "Absolute_Error": abs_err_mev,
                        "Percentage_Error": pct_err,
                        "Pull_sigma": pull,
                        "Chi2_contrib": pull**2,
                        "MSE": mse,
                    }
                )

        obs_csv = os.path.join(results_dir, f"{sector_name}_observables.csv")
        if os.path.exists(obs_csv):
            df_obs = pd.read_csv(obs_csv)
            for _, row in df_obs.iterrows():
                state_full = row["State"]
                state_label = state_full.split()[0]
                obs_type = row["Observable_Type"]
                calc_val = float(row["Value_keV"])
                obs_unc = float(row.get("Error_keV", 0.0))

                exp_val = None
                if "e+e-" in obs_type:
                    db_key = f"{sector_id}_widths_ee_keV"
                    exp_val = pdg_data.get(db_key, {}).get(state_label)
                elif "γγ" in obs_type or "gamma" in obs_type:
                    db_key = f"{sector_id}_widths_gammagamma_keV"
                    exp_val = pdg_data.get(db_key, {}).get(state_label)

                if exp_val is not None and exp_val > 0:
                    abs_err = abs(calc_val - exp_val)
                    pct_err = (abs_err / exp_val) * 100.0
                    mse = abs_err**2
                    # Pull uses the propagated width uncertainty (keV) with a small floor
                    pull = abs_err / max(obs_unc, SIGMA_FLOOR)

                    report_data.append(
                        {
                            "Sector": sector_id.upper(),
                            "Property": obs_type,
                            "State": state_full,
                            "Calculated": calc_val,
                            "Experimental": exp_val,
                            "Uncertainty": obs_unc,
                            "Absolute_Error": abs_err,
                            "Percentage_Error": pct_err,
                            "Pull_sigma": pull,
                            "Chi2_contrib": pull**2,
                            "MSE": mse,
                        }
                    )

    if report_data:
        df_report = pd.DataFrame(report_data)
        report_path = os.path.join(results_dir, "consolidated_error_report.csv")
        df_report.to_csv(report_path, index=False)
        print(f"\nConsolidated error report generated and saved to {report_path}")
