#!/usr/bin/env python3

import os
import sys
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quarkonia.gem_solver import QuarkoniumSystem, solve_gem

# --- NEW: Import the GEM parameter exporter ---
from quarkonia.metrics import (
    format_and_evaluate,
    export_gem_parameters,
    export_observables,
)
from quarkonia.observables import (
    get_mass,
    check_virial_theorem,
)
from quarkonia.decay_models import (
    get_leptonic_width,
    get_two_photon_width,
    get_3p0_decay_width,
    tune_gamma_3p0,
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

    print("\n--- Fitting/Loading Bottomonium Parameters ---")
    (bb_alpha_s, bb_b, bb_c), bb_errs = get_or_fit_parameters(
        m_1=4.730,
        m_2=4.730,
        pdg_data=all_pdg.get("bb", {}),
        r=r,
        initial_guesses=[0.350, 0.193, 0.030],
        csv_path=os.path.join(results_dir, "bb_params.csv"),
    )
    bb_sys = QuarkoniumSystem(  # sigma_smear will be calculated dynamically
        m_1=4.730, m_2=4.730, alpha_s=bb_alpha_s, b=bb_b, c=bb_c
    )
    bb_masses, bb_evecs, bb_nu = generate_spectrum(
        bb_sys,
        r,
        all_pdg.get("bb", {}),
        "Bottomonium (b_bbar)",
        bb_names,
        params_err=bb_errs,
    )

    print("\n--- Fitting/Loading Charmonium Parameters ---")
    (cc_alpha_s, cc_b, cc_c), cc_errs = get_or_fit_parameters(
        m_1=1.500,  # Charm quark mass
        m_2=1.500,  # Charm quark mass
        pdg_data=all_pdg.get("cc", {}),
        r=r,
        initial_guesses=[0.400, 0.183, -0.250],
        csv_path=os.path.join(results_dir, "cc_params.csv"),
    )
    cc_sys = QuarkoniumSystem(  # sigma_smear will be calculated dynamically
        m_1=1.500, m_2=1.500, alpha_s=cc_alpha_s, b=cc_b, c=cc_c
    )
    cc_masses, cc_evecs, cc_nu = generate_spectrum(
        cc_sys,
        r,
        all_pdg.get("cc", {}),
        "Charmonium (c_cbar)",
        cc_names,
        params_err=cc_errs,
    )

    print("\n--- Fitting/Loading B_c Meson Parameters ---")
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

    (bc_alpha_s, bc_b, bc_c), bc_errs = get_or_fit_parameters(
        m_1=4.730,
        m_2=1.500,
        pdg_data=all_pdg.get("bc", {}),
        r=r,
        initial_guesses=[bc_alpha_s_qft, 0.183, -0.090],
        csv_path=os.path.join(results_dir, "bc_params.csv"),
        bounds=([bc_alpha_s_qft - 1e-5, 0.1, -1.0], [bc_alpha_s_qft + 1e-5, 0.35, 1.0]),
    )
    bc_sys = QuarkoniumSystem(  # sigma_smear will be calculated dynamically
        m_1=4.730, m_2=1.500, alpha_s=bc_alpha_s, b=bc_b, c=bc_c
    )
    bc_masses, bc_evecs, bc_nu = generate_spectrum(
        bc_sys,
        r,
        all_pdg.get("bc", {}),
        "B_c Meson (b_cbar)",
        bc_names,
        params_err=bc_errs,
    )

    # =========================================================================
    # HADRONIC DECAY SHOWCASE: psi(3770) -> D + Dbar using 3P0 Vacuum Creation
    # =========================================================================
    print("\n--- Fitting/Loading D Meson (c_ubar) Parameters for Hadronic Decays ---")
    m_u = 0.330  # Constituent light quark mass in GeV
    (cu_alpha_s, cu_b, cu_c), cu_errs = get_or_fit_parameters(
        m_1=1.500,
        m_2=m_u,
        pdg_data={"(1^1S)": 1.864},  # D0 mass
        r=r,
        initial_guesses=[0.500, 0.180, -0.400],
        csv_path=os.path.join(results_dir, "cu_params.csv"),
        bounds=([0.2, 0.1, -1.0], [0.8, 0.3, 1.0]),
    )
    cu_sys = QuarkoniumSystem(m_1=1.500, m_2=m_u, alpha_s=cu_alpha_s, b=cu_b, c=cu_c)
    cu_names = {"1S": "D", "3S": "D^*"}

    cu_masses, cu_evecs, cu_nu = generate_spectrum(
        cu_sys,
        r,
        {"(1^1S)": 1.864},
        "D Meson (c_ubar)",
        cu_names,
        params_err=cu_errs,
        max_n=1,
        max_l=0,
    )

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
            print("-" * 80)
            print(
                f"Predicting Higher-Order Hadronic Decay: {psi_excited} -> {D_name} + {D_name}bar"
            )
            print(f"Mass {psi_excited}: {mass_psi_exc:.4f} GeV")
            if mass_psi_exc > 2.0 * mass_D:
                print(
                    f"Predicted Width (using tuned gamma={gamma_tuned:.4f}): {width_pred:.2f} MeV"
                )
            else:
                print("Decay is kinematically forbidden.")
        print("=" * 80 + "\n")
