import csv
import os
import pandas as pd
import numpy as np

from . import paths
from .fitter import SIGMA_THEORY_MEV, WIDTH_THEORY_FRAC


def format_and_evaluate(
    calculated_states, pdg_data, sector_name, sector_id, pdg_err=None, n_fit_params=4
):
    """
    Tabulate calculated vs experimental masses and report goodness-of-fit.

    The figure of merit is the RMS mass deviation (model-independent). The
    chi-square uses a *physical* per-state uncertainty:

        sigma_i = quad( sigma_exp_i , SIGMA_THEORY_MEV )

    i.e. the (tiny) PDG experimental error combined in quadrature with the
    intrinsic theory systematic of the potential model. This is what makes
    chi^2/dof interpretable: a sector that the model describes to within its
    systematic gives chi^2/dof ~ 1; charmonium, with larger relativistic
    corrections, lands above 1, which is physically meaningful rather than an
    artifact of inflated error bars.

    Parameters
    ----------
    calculated_states : dict
        Maps state name -> (mass_GeV, mass_err_GeV) or bare mass_GeV. ``mass_err``
        is the propagated *model* (parameter-covariance) uncertainty, reported for
        information; it is NOT the chi-square denominator.
    pdg_data : dict
        Experimental masses keyed by bare state label.
    sector_name : str
        Display name, used in printouts.
    sector_id : str
        Short id (bb/cc/bc/cu), used for the output path.
    pdg_err : dict, optional
        Experimental 1-sigma mass errors (GeV) keyed by bare state label.
    n_fit_params : int, optional
        Free parameters, for dof = N_states - n_fit_params.
    """
    print(f"\n--- Error Analysis vs Experimental Data ({sector_name}) ---")
    print(
        f"{'State':<18} | {'Calculated [GeV]':<25} | {'Experimental [GeV]':<18} | {'Abs Error [MeV]':<15} | {'% Error':<10} | {'Pull (sigma)':<12} | {'MSE [MeV^2]':<15}"
    )
    print("-" * 130)

    pdg_err = pdg_err or {}
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
            # Physical 1-sigma: experimental error (quadrature) with the theory floor.
            sigma_exp_mev = pdg_err.get(pure_state, 0.0) * 1000.0
            sigma_mev = np.hypot(sigma_exp_mev, SIGMA_THEORY_MEV)
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

    dof = n_compared - n_fit_params
    rmse = np.sqrt(sq_err_sum / n_compared) if n_compared else 0.0
    red_chi2 = chi2 / dof if dof > 0 else float("nan")
    print("-" * 130)
    dof_note = "" if dof > 0 else "  (under-determined: dof <= 0, reduced chi^2 undefined)"
    print(
        f"Goodness of fit ({sector_name}): N = {n_compared} states, "
        f"dof = {n_compared} - {n_fit_params} = {dof}{dof_note}"
    )
    print(
        f"   chi^2 = {chi2:.2f}   chi^2/dof = {red_chi2:.3f}   "
        f"RMS mass deviation = {rmse:.2f} MeV   "
        f"(sigma_theory = {SIGMA_THEORY_MEV:.0f} MeV)"
    )

    with open(paths.errors_csv(sector_id), "w", newline="") as f:
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
        "dof": dof,
        "chi2_per_dof": red_chi2,
        "rms_mev": rmse,
        "n": n_compared,
    }


def export_gem_parameters(nu_array, evecs, l_str, sector_id):
    """Export the compact 25-row analytical GEM coefficients for a wave."""
    data_gem = {
        "Index": np.arange(len(nu_array)),
        "nu_width": nu_array,
        f"c_1{l_str}": evecs[:, 0],
        f"c_2{l_str}": evecs[:, 1] if evecs.shape[1] > 1 else np.zeros_like(nu_array),
        f"c_3{l_str}": evecs[:, 2] if evecs.shape[1] > 2 else np.zeros_like(nu_array),
    }
    pd.DataFrame(data_gem).to_csv(paths.gem_csv(sector_id, l_str), index=False)


def export_observables(calculated_observables, sector_id):
    """Export the computed decay observables to CSV."""
    with open(paths.observables_csv(sector_id), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["State", "Observable_Type", "Value_keV", "Error_keV"])
        for state, (obs_type, obs_val, obs_err) in calculated_observables.items():
            writer.writerow([state, obs_type, f"{obs_val:.4f}", f"{obs_err:.4f}"])


def generate_consolidated_report():
    """
    Cross-reference the per-sector error/observable CSVs with the PDG data to
    build results/summary/consolidated_report.csv.

    Mass pulls are read straight from the per-sector errors.csv (already computed
    with the physical theory-floor sigma). Width pulls fold the same physical
    error model in: sigma = quad(model_error, WIDTH_THEORY_FRAC * experiment), so
    a near-zero propagated model error on a tiny D-wave width can no longer
    manufacture a spurious 100-sigma tension.
    """
    from .pdg_loader import load_pdg_data

    pdg_data = load_pdg_data()

    sectors = [
        ("bb", "Bottomonium (b_bbar)"),
        ("cc", "Charmonium (c_cbar)"),
        ("bc", "B_c Meson (b_cbar)"),
        ("cu", "D Meson (c_ubar)"),
    ]

    report_data = []

    for sector_id, sector_name in sectors:
        mass_csv = paths.errors_csv(sector_id)
        if os.path.exists(mass_csv):
            df_mass = pd.read_csv(mass_csv)
            df_valid = df_mass[df_mass["Exp_GeV"] != "N/A"].copy()
            for _, row in df_valid.iterrows():
                pull = float(row["Pull_sigma"])
                report_data.append(
                    {
                        "Sector": sector_id.upper(),
                        "Property": "Mass (GeV)",
                        "State": row["State"],
                        "Calculated": float(row["Calculated_GeV"]),
                        "Experimental": float(row["Exp_GeV"]),
                        "Uncertainty": float(row.get("Mass_Err_GeV", 0.0)) * 1000.0,
                        "Absolute_Error": float(row["Abs_Err_MeV"]),
                        "Percentage_Error": float(row["Pct_Err"]),
                        "Pull_sigma": pull,
                        "Chi2_contrib": pull**2,
                        "MSE": float(row["MSE_MeV2"]),
                    }
                )

        obs_csv = paths.observables_csv(sector_id)
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
                    exp_val = pdg_data.get(f"{sector_id}_widths_ee_keV", {}).get(state_label)
                elif "γγ" in obs_type or "gamma" in obs_type:
                    exp_val = pdg_data.get(f"{sector_id}_widths_gammagamma_keV", {}).get(state_label)

                if exp_val is not None and exp_val > 0:
                    abs_err = abs(calc_val - exp_val)
                    pct_err = (abs_err / exp_val) * 100.0
                    mse = abs_err**2
                    # Physical width sigma: model error with a ~30% NR-formula floor.
                    sigma = np.hypot(obs_unc, WIDTH_THEORY_FRAC * exp_val)
                    pull = abs_err / sigma

                    report_data.append(
                        {
                            "Sector": sector_id.upper(),
                            "Property": obs_type,
                            "State": state_full,
                            "Calculated": calc_val,
                            "Experimental": exp_val,
                            "Uncertainty": sigma,
                            "Absolute_Error": abs_err,
                            "Percentage_Error": pct_err,
                            "Pull_sigma": pull,
                            "Chi2_contrib": pull**2,
                            "MSE": mse,
                        }
                    )

    if report_data:
        df_report = pd.DataFrame(report_data)
        report_path = paths.summary_csv("consolidated_report.csv")
        df_report.to_csv(report_path, index=False)
        print(f"\nConsolidated error report generated and saved to {report_path}")
