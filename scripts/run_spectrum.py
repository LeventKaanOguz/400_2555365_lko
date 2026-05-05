#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quarkonia.gem_solver import QuarkoniumSystem, solve_gem

# --- NEW: Import the GEM parameter exporter ---
from quarkonia.metrics import (
    format_and_evaluate,
    export_gem_parameters,
    export_observables,
    generate_consolidated_report,
)
from quarkonia.observables import (
    get_mass,
    check_virial_theorem,
    calc_rms_radius,
)
from quarkonia.decay_models import (
    get_leptonic_width,
    get_two_photon_width,
    get_3p0_decay_width,
    tune_gamma_3p0,
    get_m1_decay_width,
    get_e1_decay_width,
)
from quarkonia.fitter import get_or_fit_parameters


def propagate_uncertainty(
    sys_obj, r, params_err, state_idx, spin, l, j, e_q, obs_type=None
):
    """Propagates fitter parameter uncertainty to output observables via localized finite differences."""
    eps = 1e-4
    dm_dp = []
    dobs_dp = []
    for p_idx in range(3):
        p_up, p_dn = (
            [sys_obj.alpha_s, sys_obj.b, sys_obj.c],
            [sys_obj.alpha_s, sys_obj.b, sys_obj.c],
        )
        p_up[p_idx] += eps
        p_dn[p_idx] -= eps

        sys_up = QuarkoniumSystem(sys_obj.m_1, sys_obj.m_2, *p_up)
        sys_dn = QuarkoniumSystem(sys_obj.m_1, sys_obj.m_2, *p_dn)

        m_up = get_mass(
            solve_gem(sys_up, r, l, spin)[0],
            solve_gem(sys_up, r, l, spin)[2],
            solve_gem(sys_up, r, l, spin)[3],
            sys_up,
            state_idx,
            spin,
            l,
            j,
        )
        m_dn = get_mass(
            solve_gem(sys_dn, r, l, spin)[0],
            solve_gem(sys_dn, r, l, spin)[2],
            solve_gem(sys_dn, r, l, spin)[3],
            sys_dn,
            state_idx,
            spin,
            l,
            j,
        )
        dm_dp.append(((m_up - m_dn) / (2 * eps)) * params_err[p_idx])

        if obs_type == "leptonic":
            w_up = get_leptonic_width(
                m_up,
                solve_gem(sys_up, r, l, spin)[2][:, state_idx],
                solve_gem(sys_up, r, l, spin)[3],
                sys_up,
                e_q,
                l=l,
            )
            w_dn = get_leptonic_width(
                m_dn,
                solve_gem(sys_dn, r, l, spin)[2][:, state_idx],
                solve_gem(sys_dn, r, l, spin)[3],
                sys_dn,
                e_q,
                l=l,
            )
            dobs_dp.append(((w_up - w_dn) / (2 * eps)) * params_err[p_idx])
        elif obs_type == "two_photon":
            w_up = get_two_photon_width(
                m_up,
                solve_gem(sys_up, r, l, spin)[2][:, state_idx],
                solve_gem(sys_up, r, l, spin)[3],
                sys_up,
                e_q,
                l=l,
            )
            w_dn = get_two_photon_width(
                m_dn,
                solve_gem(sys_dn, r, l, spin)[2][:, state_idx],
                solve_gem(sys_dn, r, l, spin)[3],
                sys_dn,
                e_q,
                l=l,
            )
            dobs_dp.append(((w_up - w_dn) / (2 * eps)) * params_err[p_idx])

    return np.sqrt(np.sum(np.array(dm_dp) ** 2)), (
        np.sqrt(np.sum(np.array(dobs_dp) ** 2)) if obs_type else 0.0
    )


def parse_state_name(state_name):
    name_part = state_name.split()[0]
    n = int(name_part[1])
    spin = 1 if name_part[3] == "3" else 0
    l_char = name_part[4]
    l = {"S": 0, "P": 1, "D": 2, "F": 3}[l_char]
    if "_" in name_part:
        j = int(name_part.split("_")[1].replace(")", ""))
    else:
        if l == 0:
            j = 1 if spin == 1 else 0
        else:
            j = l
    return l, spin, n - 1, j


def propagate_transition_uncertainty(
    sys_obj,
    r,
    params_err,
    state_i_name,
    state_f_name,
    decay_type,
    e_q=0.0,
    gamma_3p0=0.4,
    m_B=None,
    evec_B=None,
    nu_B=None,
    m_C=None,
    evec_C=None,
    nu_C=None,
):
    eps = 1e-4
    dobs_dp = []

    l_i, spin_i, idx_i, j_i = parse_state_name(state_i_name)
    if decay_type in ["M1", "E1"]:
        l_f, spin_f, idx_f, j_f = parse_state_name(state_f_name)

    for p_idx in range(3):
        if params_err[p_idx] == 0.0:
            dobs_dp.append(0.0)
            continue

        p_up = [sys_obj.alpha_s, sys_obj.b, sys_obj.c]
        p_dn = [sys_obj.alpha_s, sys_obj.b, sys_obj.c]

        p_up[p_idx] += eps
        p_dn[p_idx] -= eps

        sys_up = QuarkoniumSystem(sys_obj.m_1, sys_obj.m_2, *p_up)
        sys_dn = QuarkoniumSystem(sys_obj.m_1, sys_obj.m_2, *p_dn)

        # solve up
        evals_i_up, _, evecs_i_up, nu_i_up = solve_gem(sys_up, r, l=l_i, spin=spin_i)
        m_i_up = get_mass(
            evals_i_up, evecs_i_up, nu_i_up, sys_up, idx_i, spin_i, l_i, j_i
        )

        # solve dn
        evals_i_dn, _, evecs_i_dn, nu_i_dn = solve_gem(sys_dn, r, l=l_i, spin=spin_i)
        m_i_dn = get_mass(
            evals_i_dn, evecs_i_dn, nu_i_dn, sys_dn, idx_i, spin_i, l_i, j_i
        )

        if decay_type in ["M1", "E1"]:
            evals_f_up, _, evecs_f_up, nu_f_up = solve_gem(
                sys_up, r, l=l_f, spin=spin_f
            )
            m_f_up = get_mass(
                evals_f_up, evecs_f_up, nu_f_up, sys_up, idx_f, spin_f, l_f, j_f
            )

            evals_f_dn, _, evecs_f_dn, nu_f_dn = solve_gem(
                sys_dn, r, l=l_f, spin=spin_f
            )
            m_f_dn = get_mass(
                evals_f_dn, evecs_f_dn, nu_f_dn, sys_dn, idx_f, spin_f, l_f, j_f
            )

            if decay_type == "M1":
                w_up = get_m1_decay_width(
                    m_i_up,
                    m_f_up,
                    evecs_i_up[:, idx_i],
                    nu_i_up,
                    evecs_f_up[:, idx_f],
                    nu_f_up,
                    sys_up,
                    e_q,
                )
                w_dn = get_m1_decay_width(
                    m_i_dn,
                    m_f_dn,
                    evecs_i_dn[:, idx_i],
                    nu_i_dn,
                    evecs_f_dn[:, idx_f],
                    nu_f_dn,
                    sys_dn,
                    e_q,
                )
            else:
                w_up = get_e1_decay_width(
                    m_i_up,
                    m_f_up,
                    evecs_i_up[:, idx_i],
                    nu_i_up,
                    evecs_f_up[:, idx_f],
                    nu_f_up,
                    sys_up,
                    e_q,
                )
                w_dn = get_e1_decay_width(
                    m_i_dn,
                    m_f_dn,
                    evecs_i_dn[:, idx_i],
                    nu_i_dn,
                    evecs_f_dn[:, idx_f],
                    nu_f_dn,
                    sys_dn,
                    e_q,
                )
        elif decay_type == "3P0":
            w_up = get_3p0_decay_width(
                m_i_up,
                m_B,
                m_C,
                evecs_i_up[:, idx_i],
                nu_i_up,
                evec_B,
                nu_B,
                evec_C,
                nu_C,
                l_A=l_i,
                gamma_3p0=gamma_3p0,
            )
            w_dn = get_3p0_decay_width(
                m_i_dn,
                m_B,
                m_C,
                evecs_i_dn[:, idx_i],
                nu_i_dn,
                evec_B,
                nu_B,
                evec_C,
                nu_C,
                l_A=l_i,
                gamma_3p0=gamma_3p0,
            )

        dobs_dp.append(((w_up - w_dn) / (2 * eps)) * params_err[p_idx])

    return np.sqrt(np.sum(np.array(dobs_dp) ** 2))


def generate_spectrum(
    sys_obj: QuarkoniumSystem,
    r,
    pdg_data,
    sector_name,
    particle_names,
    params_err=None,
    max_n=3,
    max_l=2,
):
    """
    Dynamically generates the full spectroscopic multiplet n^(2S+1)L_J
    and maps them to standard meson names.

    Parameters
    ----------
    sys_obj : QuarkoniumSystem
        Quarkonium system parameters object.
    r : numpy.ndarray
        Spatial grid arrays mapping distances.
    pdg_data : dict
        Experimental PDG mass data.
    sector_name : str
        Particle sector designation (e.g. 'Charmonium').
    particle_names : dict
        Dictionary mapping standard family symbols to specific states.
    max_n : int, optional
        Maximum primary quantum number limit, by default 3.
    max_l : int, optional
        Maximum orbital angular momentum limit, by default 2.
    """
    calculated_masses = {}
    calculated_wavefuncs = {}
    calculated_evecs = {}
    calculated_observables = {}

    l_chars = {0: "S", 1: "P", 2: "D", 3: "F"}

    # Charge assignment for EM annihilation decays
    e_q = 0.0
    if "b_bbar" in sector_name:
        e_q = -1.0 / 3.0
    elif "c_cbar" in sector_name:
        e_q = 2.0 / 3.0
    # Note: b_cbar (B_c meson) is excluded because it is a charged, mixed-flavor
    # meson. It cannot annihilate into e+e- via a virtual photon. Its leptonic
    # decays proceed via the weak interaction (W boson -> l + nu).

    print(f"\nSolving GEM eigenstates for {sector_name}...")
    for l in range(max_l + 1):
        for spin in [0, 1]:
            # Solve the analytical GEM eigensystem for each independent (L, S) channel
            evals, u_gem, evecs, nu_array = solve_gem(sys_obj, r, l=l, spin=spin)

            # Export the representative Singlet state matrix metrics
            if spin == 0:
                export_gem_parameters(nu_array, evecs, l_chars[l], sector_name)
                if l == 0:
                    virial_ratio = (
                        check_virial_theorem(evecs[:, 0], nu_array, sys_obj, l=0)
                        if len(evals) > 0
                        else 0.0
                    )
                    print(
                        f"\n--- GEM Optimized Parameters (S-Wave Singlet Basis) [Virial Ratio: {virial_ratio:.5f}] ---"
                    )
                    print(
                        f"{'Index':<7} | {'nu (width)':<15} | {'c_1S (Ground)':<18} | {'c_2S (1st Excited)'}"
                    )
                    print("-" * 67)
                    for i in range(len(nu_array)):
                        c1s = evecs[i, 0] if len(evals) > 0 else 0.0
                        c2s = evecs[i, 1] if len(evals) > 1 else 0.0
                        print(
                            f"{i:<7} | {nu_array[i]:<15.6e} | {c1s:<18.6e} | {c2s:.6e}"
                        )
                    print()

            j_list = [l] if spin == 0 else list(range(abs(l - 1), l + 2))

            for j in j_list:
                for n in range(1, max_n + 1):
                    state_idx = n - 1
                    if state_idx >= len(evals):
                        continue

                    spin_str = "1" if spin == 0 else "3"
                    l_str = l_chars[l]
                    family = f"{spin_str}{l_str}"
                    symbol = particle_names.get(family, "")

                    if l == 0:
                        name = f"({n}^{spin_str}S) {symbol}"
                    elif spin == 0:
                        name = f"({n}^{spin_str}{l_str}) {symbol}"
                    else:
                        name = f"({n}^{spin_str}{l_str}_{j}) {symbol}"

                    if name == "(1^3S) ψ":
                        name = "(1^3S) J/ψ"
                    if name == "(1^3S) Υ":
                        name = "(1^3S) Υ_b"

                    mass = get_mass(
                        evals, evecs, nu_array, sys_obj, state_idx, spin, l, j
                    )

                    mass_err = 0.0
                    if params_err is not None and sum(params_err) > 0:
                        mass_err, _ = propagate_uncertainty(
                            sys_obj, r, params_err, state_idx, spin, l, j, e_q, None
                        )

                    calculated_masses[name] = (mass, mass_err)
                    calculated_wavefuncs[name] = u_gem[:, state_idx]
                    calculated_evecs[name] = evecs[:, state_idx]

                    # RMS Radius Calculation
                    rms_rad = calc_rms_radius(evecs[:, state_idx], nu_array, l=l)
                    calculated_observables[name + "_rms"] = (
                        "RMS Radius (GeV^-1)",
                        rms_rad,
                        0.0,
                    )
                    # --- Compute Decay Observables ---
                    if e_q != 0.0:
                        if spin == 1:
                            width_ee = get_leptonic_width(
                                mass,
                                evecs[:, state_idx],
                                nu_array,
                                sys_obj,
                                e_q,
                                l=l,
                            )
                            obs_err = 0.0
                            if params_err is not None and sum(params_err) > 0:
                                _, obs_err = propagate_uncertainty(
                                    sys_obj,
                                    r,
                                    params_err,
                                    state_idx,
                                    spin,
                                    l,
                                    j,
                                    e_q,
                                    "leptonic",
                                )
                            calculated_observables[name] = (
                                "Leptonic Width (e+e-)",
                                width_ee,
                                obs_err,
                            )
                        elif spin == 0:
                            width_gg = get_two_photon_width(
                                mass,
                                evecs[:, state_idx],
                                nu_array,
                                sys_obj,
                                e_q,
                                l=l,
                            )
                            obs_err = 0.0
                            if params_err is not None and sum(params_err) > 0:
                                _, obs_err = propagate_uncertainty(
                                    sys_obj,
                                    r,
                                    params_err,
                                    state_idx,
                                    spin,
                                    l,
                                    j,
                                    e_q,
                                    "two_photon",
                                )
                            calculated_observables[name] = (
                                "Two-Photon Width (γγ)",
                                width_gg,
                                obs_err,
                            )

    format_and_evaluate(calculated_masses, pdg_data, sector_name)

    if calculated_observables:
        print(f"\n--- Decay Observables for {sector_name} ---")
        for state, (obs_type, obs_val, obs_err) in calculated_observables.items():
            print(f"{state:<15} | {obs_type}: {obs_val:.2f} ± {obs_err:.2f} keV")

        export_observables(calculated_observables, sector_name)

    return calculated_masses, calculated_evecs, nu_array


if __name__ == "__main__":
    pdg_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "pdg_data.json")
    )
    with open(pdg_path, "r") as f:
        all_pdg = json.load(f)

    R_max = 15.0
    N = 6000
    dr = R_max / N
    r = np.linspace(dr, R_max, N)

    bb_names = {
        "1S": "η_b",
        "3S": "Υ",
        "1P": "h_b",
        "3P": "χ_b",
        "1D": "η_b2",
        "3D": "Υ",
    }
    cc_names = {
        "1S": "η_c",
        "3S": "ψ",
        "1P": "h_c",
        "3P": "χ_c",
        "1D": "η_c2",
        "3D": "ψ",
    }
    bc_names = {
        "1S": "B_c",
        "3S": "B_c^*",
        "1P": "B_{c1}",
        "3P": "B_{c0,1,2}^*",
        "1D": "B_{c2}",
        "3D": "B_{c1,2,3}^*",
    }

    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )

    # We need to get bb and cc alpha_s first to do the B_c interpolation
    (cc_alpha_s, _, _), _ = get_or_fit_parameters(
        1.500,
        1.500,
        all_pdg.get("cc", {}),
        r,
        [0.400, 0.183, -0.250],
        os.path.join(results_dir, "cc_params.csv"),
    )
    (bb_alpha_s, _, _), _ = get_or_fit_parameters(
        4.730,
        4.730,
        all_pdg.get("bb", {}),
        r,
        [0.350, 0.193, 0.030],
        os.path.join(results_dir, "bb_params.csv"),
    )

    # QFT Logarithmic Interpolation of alpha_s based on reduced mass scale
    mu_cc = 1.500 / 2.0
    mu_bb = 4.730 / 2.0
    mu_bc = (4.730 * 1.500) / (4.730 + 1.500)

    inv_alpha_cc = 1.0 / cc_alpha_s
    inv_alpha_bb = 1.0 / bb_alpha_s
    inv_alpha_bc = inv_alpha_cc + (np.log(mu_bc / mu_cc) / np.log(mu_bb / mu_cc)) * (
        inv_alpha_bb - inv_alpha_cc
    )
    bc_alpha_s_qft = 1.0 / inv_alpha_bc

    m_u = 0.330  # Constituent light quark mass in GeV
    cu_names = {"1S": "D", "3S": "D^*"}

    sectors_config = [
        {
            "id": "bb",
            "name": "Bottomonium (b_bbar)",
            "m_1": 4.730,
            "m_2": 4.730,
            "pdg_data": all_pdg.get("bb", {}),
            "names": bb_names,
            "initial_guesses": [0.350, 0.193, 0.030],
            "bounds": None,
            "max_n": 3,
            "max_l": 2,
        },
        {
            "id": "cc",
            "name": "Charmonium (c_cbar)",
            "m_1": 1.500,
            "m_2": 1.500,
            "pdg_data": all_pdg.get("cc", {}),
            "names": cc_names,
            "initial_guesses": [0.400, 0.183, -0.250],
            "bounds": None,
            "max_n": 3,
            "max_l": 2,
        },
        {
            "id": "bc",
            "name": "B_c Meson (b_cbar)",
            "m_1": 4.730,
            "m_2": 1.500,
            "pdg_data": all_pdg.get("bc", {}),
            "names": bc_names,
            "initial_guesses": [bc_alpha_s_qft, 0.183, -0.090],
            "bounds": (
                [bc_alpha_s_qft - 1e-5, 0.1, -1.0],
                [bc_alpha_s_qft + 1e-5, 0.35, 1.0],
            ),
            "max_n": 3,
            "max_l": 2,
        },
        {
            "id": "cu",
            "name": "D Meson (c_ubar)",
            "m_1": 1.500,
            "m_2": m_u,
            "pdg_data": {"(1^1S)": 1.864},
            "names": cu_names,
            "initial_guesses": [0.500, 0.180, -0.400],
            "bounds": ([0.2, 0.1, -1.0], [0.8, 0.3, 1.0]),
            "max_n": 1,
            "max_l": 0,
        },
    ]

    results_dict = {}

    for config in sectors_config:
        print(f"\n--- Fitting/Loading {config['name']} Parameters ---")
        (alpha_s, b, c), errs = get_or_fit_parameters(
            m_1=config["m_1"],
            m_2=config["m_2"],
            pdg_data=config["pdg_data"],
            r=r,
            initial_guesses=config["initial_guesses"],
            csv_path=os.path.join(results_dir, f"{config['id']}_params.csv"),
            bounds=config["bounds"],
        )
        sys_obj = QuarkoniumSystem(
            m_1=config["m_1"], m_2=config["m_2"], alpha_s=alpha_s, b=b, c=c
        )
        masses, evecs, nu = generate_spectrum(
            sys_obj,
            r,
            config["pdg_data"],
            config["name"],
            config["names"],
            params_err=errs,
            max_n=config["max_n"],
            max_l=config["max_l"],
        )
        results_dict[config["id"]] = {
            "masses": masses,
            "evecs": evecs,
            "nu": nu,
            "sys_obj": sys_obj,
            "errs": errs,
        }

    # =========================================================================
    # HADRONIC DECAY SHOWCASE: psi(3770) -> D + Dbar using 3P0 Vacuum Creation
    # =========================================================================
    cc_masses = results_dict["cc"]["masses"]
    cc_evecs = results_dict["cc"]["evecs"]
    cc_nu = results_dict["cc"]["nu"]
    cu_masses = results_dict["cu"]["masses"]
    cu_evecs = results_dict["cu"]["evecs"]
    cu_nu = results_dict["cu"]["nu"]

    psi_name = "(1^3D_1) ψ"  # D-wave charmonium psi(3770)
    D_name = "(1^1S) D"  # S-wave D meson

    if psi_name in cc_masses and D_name in cu_masses:
        mass_psi = cc_masses[psi_name][0]
        mass_D = cu_masses[D_name][0]

        c_psi = cc_evecs[psi_name]
        c_D = cu_evecs[D_name]

        target_exp_width = 27.2
        gamma_tuned = tune_gamma_3p0(
            target_width_MeV=target_exp_width,
            mass_A=mass_psi,
            mass_B=mass_D,
            mass_C=mass_D,
            c_A=c_psi,
            nu_A=cc_nu,
            c_B=c_D,
            nu_B=cu_nu,
            c_C=c_D,
            nu_C=cu_nu,
            l_A=2,  # psi(3770) is an L=2 state
            initial_gamma=0.4,
        )

        width_3p0_tuned = get_3p0_decay_width(
            mass_A=mass_psi,
            mass_B=mass_D,
            mass_C=mass_D,
            c_A=c_psi,
            nu_A=cc_nu,
            c_B=c_D,
            nu_B=cu_nu,
            c_C=c_D,
            nu_C=cu_nu,
            l_A=2,
            gamma_3p0=gamma_tuned,
        )

        print("\n" + "=" * 80)
        print(f"--- Showcase: Hadronic Decay via 3P0 Model ---")
        print(f"Transition: {psi_name} -> {D_name} + {D_name}bar")
        print(f"Mass {psi_name}: {mass_psi:.4f} GeV")
        print(f"Mass {D_name} (x2):  {mass_D * 2.0:.4f} GeV")
        if mass_psi > 2.0 * mass_D:
            print(
                f"Tuning 3P0 gamma to match experimental width: {target_exp_width} MeV"
            )
            print(f"Optimized gamma value: {gamma_tuned:.4f}")
            print(f"Calculated 3P0 Decay Width: {width_3p0_tuned:.2f} MeV")
        else:
            print("Decay is kinematically forbidden (Mass A < Mass B + Mass C).")

        # Predict an unknown decay using the locked-in tuned gamma
        psi_excited = "(2^3D_1) ψ"
        if psi_excited in cc_masses:
            mass_psi_exc = cc_masses[psi_excited][0]
            c_psi_exc = cc_evecs[psi_excited]
            width_pred = get_3p0_decay_width(
                mass_A=mass_psi_exc,
                mass_B=mass_D,
                mass_C=mass_D,
                c_A=c_psi_exc,
                nu_A=cc_nu,
                c_B=c_D,
                nu_B=cu_nu,
                c_C=c_D,
                nu_C=cu_nu,
                l_A=2,
                gamma_3p0=gamma_tuned,
            )

            width_pred_err = propagate_transition_uncertainty(
                sys_obj=results_dict["cc"]["sys_obj"],
                r=r,
                params_err=results_dict["cc"]["errs"],
                state_i_name=psi_excited,
                state_f_name=None,
                decay_type="3P0",
                gamma_3p0=gamma_tuned,
                m_B=mass_D,
                evec_B=c_D,
                nu_B=cu_nu,
                m_C=mass_D,
                evec_C=c_D,
                nu_C=cu_nu,
            )
            print("-" * 80)
            print(
                f"Predicting Higher-Order Hadronic Decay: {psi_excited} -> {D_name} + {D_name}bar"
            )
            print(f"Mass {psi_excited}: {mass_psi_exc:.4f} GeV")
            if mass_psi_exc > 2.0 * mass_D:
                print(
                    f"Predicted Width (using tuned gamma={gamma_tuned:.4f}): {width_pred:.2f} ± {width_pred_err:.2f} MeV"
                )
            else:
                print("Decay is kinematically forbidden.")

    # =========================================================================
    # RADIATIVE DECAY SHOWCASE: M1 and E1 Transitions
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- Radiative Transitions (M1 & E1) ---")

    radiative_results = []

    for sector, eq, m1_pairs, e1_pairs in [
        (
            "cc",
            2.0 / 3.0,
            [("(1^3S) J/ψ", "(1^1S) η_c"), ("(2^3S) ψ", "(2^1S) η_c")],
            [("(1^3P_0) χ_c", "(1^3S) J/ψ"), ("(1^3P_1) χ_c", "(1^3S) J/ψ")],
        ),
        (
            "bb",
            -1.0 / 3.0,
            [("(1^3S) Υ_b", "(1^1S) η_b"), ("(2^3S) Υ", "(2^1S) η_b")],
            [("(1^3P_0) χ_b", "(1^3S) Υ_b"), ("(1^3P_1) χ_b", "(1^3S) Υ_b")],
        ),
    ]:
        masses = results_dict[sector]["masses"]
        evecs = results_dict[sector]["evecs"]
        nu = results_dict[sector]["nu"]

        sys_obj = results_dict[sector]["sys_obj"]
        errs = results_dict[sector]["errs"]

        for state_i, state_f in m1_pairs:
            if state_i in masses and state_f in masses:
                m_i, _ = masses[state_i]
                m_f, _ = masses[state_f]
                w_m1 = get_m1_decay_width(
                    m_i, m_f, evecs[state_i], nu, evecs[state_f], nu, sys_obj, eq
                )
                w_m1_err = propagate_transition_uncertainty(
                    sys_obj, r, errs, state_i, state_f, "M1", e_q=eq
                )
                print(
                    f"[{sector.upper()}] M1: {state_i:<15} -> {state_f:<15} + γ : {w_m1:.3f} ± {w_m1_err:.3f} keV"
                )
                radiative_results.append(
                    {
                        "Sector": sector.upper(),
                        "Transition": f"{state_i} -> {state_f} + γ",
                        "Type": "M1",
                        "Width_keV": w_m1,
                        "Width_err_keV": w_m1_err,
                    }
                )

        for state_i, state_f in e1_pairs:
            if state_i in masses and state_f in masses:
                m_i, _ = masses[state_i]
                m_f, _ = masses[state_f]
                w_e1 = get_e1_decay_width(
                    m_i, m_f, evecs[state_i], nu, evecs[state_f], nu, sys_obj, eq
                )
                w_e1_err = propagate_transition_uncertainty(
                    sys_obj, r, errs, state_i, state_f, "E1", e_q=eq
                )
                print(
                    f"[{sector.upper()}] E1: {state_i:<15} -> {state_f:<15} + γ : {w_e1:.3f} ± {w_e1_err:.3f} keV"
                )
                radiative_results.append(
                    {
                        "Sector": sector.upper(),
                        "Transition": f"{state_i} -> {state_f} + γ",
                        "Type": "E1",
                        "Width_keV": w_e1,
                        "Width_err_keV": w_e1_err,
                    }
                )

    if radiative_results:
        rad_df = pd.DataFrame(radiative_results)
        rad_csv_path = os.path.join(results_dir, "radiative_decays.csv")
        rad_df.to_csv(rad_csv_path, index=False)
        print(f"\nExported radiative transitions to {rad_csv_path}")

    print("=" * 80 + "\n")

    generate_consolidated_report()
