import csv
import os
import pandas as pd
import numpy as np

from . import paths


def format_and_evaluate(
    calculated_states, pdg_data, sector_name, sector_id, pdg_err=None,
    n_fit_params=4, sigma_theory_mev=None, leptonic_obs=None,
):
    """
    Tabulate calculated vs experimental masses and report goodness-of-fit.

    The figure of merit is the RMS mass deviation (model-independent). The
    chi-square / pull uses the **computational** uncertainty:

        sigma_i = quad( sigma_exp_i , sigma_comp_i )

    where ``sigma_comp_i`` is the propagated parameter-covariance uncertainty on
    the predicted mass (``mass_err``, the finite-difference propagation of the
    fitted Cornell covariance done in run_spectrum.propagate_uncertainty), combined
    in quadrature with the PDG experimental error. This is the textbook pull
    denominator (parametric prediction error + experimental error): it validates
    whether the *computational implementation* reproduces experiment, not whether
    the non-relativistic *model* is physically complete.

    The relativistic-truncation theory bar ``sigma_theory_i`` is deliberately NOT
    in this denominator -- it answers a different question (is the method viable),
    so it is recorded as an informational ``Sigma_Theory_MeV`` column / figure band
    only. The propagated sigma_comp is sizable (tens of MeV), comparable to or larger
    than the mass deviations, so chi^2/dof here typically lands at or below 1: the
    implementation reproduces experiment to within its computed precision, with pulls
    mostly sub-1-sigma. Quote the RMS deviation as the headline; it assumes nothing
    about sigma.

    Parameters
    ----------
    calculated_states : dict
        Maps state name -> (mass_GeV, mass_err_GeV) or bare mass_GeV. ``mass_err``
        is the propagated computational (parameter-covariance) uncertainty; it IS
        the chi-square / pull denominator (in quadrature with the experimental error).
    pdg_data : dict
        Experimental masses keyed by bare state label.
    sector_name : str
        Display name, used in printouts.
    sector_id : str
        Short id (bb/cc/bc), used for the output path.
    pdg_err : dict, optional
        Experimental 1-sigma mass errors (GeV) keyed by bare state label.
    n_fit_params : int, optional
        Free parameters, for dof = N_states - n_fit_params.
    sigma_theory_mev : dict, optional
        Derived per-state theory sigma (MeV), keyed by full state name. Reported as
        an informational model-viability diagnostic only -- NOT the denominator.
    """
    st_theory = sigma_theory_mev or {}
    default_theory = float(np.mean(list(st_theory.values()))) if st_theory else 0.0
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

        sigma_th_mev = st_theory.get(state, default_theory)
        # Computational 1-sigma: propagated fitted-parameter covariance on this
        # predicted mass (mass_err, GeV -> MeV). This is the pull denominator.
        sigma_comp_mev = mass_err * 1000.0
        if exp_mass is not None:
            abs_err = (calc_mass - exp_mass) * 1000.0
            pct_err = abs(calc_mass - exp_mass) / exp_mass * 100.0
            mse = abs_err**2
            # Physical 1-sigma: experimental error in quadrature with the
            # propagated *computational* uncertainty (NOT the theory bar).
            sigma_exp_mev = pdg_err.get(pure_state, 0.0) * 1000.0
            sigma_mev = np.hypot(sigma_exp_mev, sigma_comp_mev)
            pull = abs_err / sigma_mev if sigma_mev > 0 else 0.0
            chi2 += pull**2
            sq_err_sum += mse
            n_compared += 1
            print(
                f"{state:<18} | {calc_str:<25} | {exp_mass:<18.4f} | {abs_err:<15.1f} | {pct_err:<10.3f} | {pull:<12.2f} | {mse:<15.2f}"
            )
            results.append(
                [state, calc_mass, mass_err, exp_mass, abs_err, pct_err, pull, mse,
                 sigma_th_mev, sigma_comp_mev]
            )
        else:
            print(
                f"{state:<18} | {calc_str:<25} | {'-':<18} | {'-':<15} | {'-':<10} | {'-':<12} | {'-':<15}"
            )
            results.append(
                [state, calc_mass, mass_err, "N/A", "N/A", "N/A", "N/A", "N/A",
                 sigma_th_mev, sigma_comp_mev]
            )

    dof = n_compared - n_fit_params
    rmse = np.sqrt(sq_err_sum / n_compared) if n_compared else 0.0
    red_chi2 = chi2 / dof if dof > 0 else float("nan")

    # Combined validation chi-square: fold the S-wave vector (^3S_1) leptonic
    # widths in alongside the masses, so the validation set matches the
    # observables the FIT actually optimizes (masses + e+e- widths) rather than
    # masses alone. Each width pull uses the SAME computational denominator as the
    # masses, quad(sigma_exp, sigma_comp) -- no new error model. The D-wave
    # leptonic widths (psi(3770)/psi(4160)), the two-photon and the radiative
    # widths are deliberately EXCLUDED: they probe model *completeness* (e.g. the
    # missing coupled-channel S-D mixing that leaves psi(3770) at ~10 sigma), which
    # sigma_comp does not test. They stay reported as pure predictions in the
    # consolidated report and the limitations note.
    chi2_lep = 0.0
    n_lep = 0
    for _lbl, g_calc, s_comp, g_exp, s_exp in (leptonic_obs or []):
        if g_exp is None or g_exp <= 0:
            continue
        sig = np.hypot(s_exp or 0.0, s_comp or 0.0)
        if sig <= 0:
            continue
        chi2_lep += ((g_calc - g_exp) / sig) ** 2
        n_lep += 1
    n_comb = n_compared + n_lep
    dof_comb = n_comb - n_fit_params
    chi2_comb = chi2 + chi2_lep
    red_chi2_comb = chi2_comb / dof_comb if dof_comb > 0 else float("nan")
    print("-" * 130)
    dof_note = "" if dof > 0 else "  (under-determined: dof <= 0, reduced chi^2 undefined)"
    print(
        f"Goodness of fit ({sector_name}): N = {n_compared} states, "
        f"dof = {n_compared} - {n_fit_params} = {dof}{dof_note}"
    )
    th_vals = list(st_theory.values())
    th_note = (
        f"(sigma_theory diagnostic, per state: {min(th_vals):.0f}-{max(th_vals):.0f} MeV"
        " -- NOT in the pull denominator)"
        if th_vals else "(pull sigma: experimental + computational)"
    )
    print(
        f"   chi^2 = {chi2:.2f}   chi^2/dof = {red_chi2:.3f}   "
        f"RMS mass deviation = {rmse:.2f} MeV   {th_note}"
    )
    if n_lep:
        print(
            f"   + {n_lep} S-wave e+e- widths -> combined chi^2 = {chi2_comb:.2f}, "
            f"dof = {dof_comb}, chi^2/dof = {red_chi2_comb:.3f}  "
            "(D-wave/gamma-gamma/radiative excluded -- pure predictions)"
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
                    "Sigma_Theory_MeV",
                    "Sigma_Comp_MeV",
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
        "n_lep": n_lep,
        "chi2_comb": chi2_comb,
        "dof_comb": dof_comb,
        "chi2_per_dof_comb": red_chi2_comb,
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
    with the propagated computational sigma). Width pulls use the same textbook
    denominator as the masses: sigma = quad(sigma_exp, sigma_comp), the experimental
    width error (read from the PDG dump by pdg_loader) in quadrature with the
    propagated parameter-covariance error on the predicted width (obs_unc). The
    relativistic/QCD theory fraction is a model-viability bar -- not a computational
    one -- so it stays excluded from the denominator (consistent with the mass pulls).
    """
    from .pdg_loader import load_pdg_data

    pdg_data = load_pdg_data()

    sectors = [
        ("bb", "Bottomonium (b_bbar)"),
        ("cc", "Charmonium (c_cbar)"),
        ("bc", "B_c Meson (b_cbar)"),
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
                        "Sigma_Theory_MeV": float(row.get("Sigma_Theory_MeV", 0.0)),
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

                exp_val = exp_err = None
                if "e+e-" in obs_type:
                    exp_val = pdg_data.get(f"{sector_id}_widths_ee_keV", {}).get(state_label)
                    exp_err = pdg_data.get(f"{sector_id}_widths_ee_err_keV", {}).get(state_label)
                elif "γγ" in obs_type or "gamma" in obs_type:
                    exp_val = pdg_data.get(f"{sector_id}_widths_gammagamma_keV", {}).get(state_label)
                    exp_err = pdg_data.get(f"{sector_id}_widths_gammagamma_err_keV", {}).get(state_label)

                if exp_val is not None and exp_val > 0:
                    abs_err = abs(calc_val - exp_val)
                    pct_err = (abs_err / exp_val) * 100.0
                    mse = abs_err**2
                    # Width pull sigma: experimental width error (from the PDG dump,
                    # via pdg_loader) in quadrature with the propagated parameter-
                    # covariance error on the predicted width (obs_unc). Same textbook
                    # denominator as the masses -- quad(sigma_exp, sigma_comp), no theory
                    # term. The experimental error matters here because sigma_comp is a
                    # small fraction of the width, so omitting it inflated the pulls of
                    # the near-zero D-wave leptonic widths into the hundreds.
                    sigma = np.hypot(exp_err or 0.0, obs_unc)
                    pull = abs_err / sigma if sigma > 0 else 0.0

                    report_data.append(
                        {
                            "Sector": sector_id.upper(),
                            "Property": obs_type,
                            "State": state_full,
                            "Calculated": calc_val,
                            "Experimental": exp_val,
                            # Reported +/- is the propagated computational sigma (matches
                            # the D-figure error bars); the pull below divides by the
                            # full quad(sigma_exp, sigma_comp).
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
        report_path = paths.summary_csv("consolidated_report.csv")
        df_report.to_csv(report_path, index=False)
        print(f"\nConsolidated error report generated and saved to {report_path}")
