#!/usr/bin/env python3

import os
import sys
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
    calc_relativistic_shift,
    calc_tensor_mixing_exact,
)
from quarkonia.decay_models import (
    get_leptonic_width,
    get_leptonic_width_mixed,
    get_two_photon_width,
    get_two_photon_width_pwave,
    get_m1_decay_width,
    get_e1_decay_width,
)
from quarkonia.fitter import get_or_fit_parameters
from quarkonia.pdg_loader import load_pdg_data
from quarkonia import paths


def propagate_uncertainty(
    sys_obj, r, params_err, state_idx, spin, l, j, e_q, obs_type=None
):
    """Propagates fitter parameter uncertainty to output observables via localized finite differences."""
    eps = 1e-4
    dm_dp = []
    dobs_dp = []
    for p_idx in range(4):
        p_up = [sys_obj.alpha_s, sys_obj.b, sys_obj.c, sys_obj.sigma_smear]
        p_dn = [sys_obj.alpha_s, sys_obj.b, sys_obj.c, sys_obj.sigma_smear]
        p_up[p_idx] += eps
        p_dn[p_idx] -= eps

        sys_up = QuarkoniumSystem(sys_obj.m_1, sys_obj.m_2, *p_up)
        sys_dn = QuarkoniumSystem(sys_obj.m_1, sys_obj.m_2, *p_dn)

        # Solve once per (up/dn) and reuse for all observables
        ev_up, _, evec_up, nu_up = solve_gem(sys_up, r, l, spin)
        ev_dn, _, evec_dn, nu_dn = solve_gem(sys_dn, r, l, spin)

        m_up = get_mass(ev_up, evec_up, nu_up, sys_up, state_idx, spin, l, j)
        m_dn = get_mass(ev_dn, evec_dn, nu_dn, sys_dn, state_idx, spin, l, j)
        dm_dp.append(((m_up - m_dn) / (2 * eps)) * params_err[p_idx])

        if obs_type == "leptonic":
            w_up = get_leptonic_width(m_up, evec_up[:, state_idx], nu_up, sys_up, e_q, l=l)
            w_dn = get_leptonic_width(m_dn, evec_dn[:, state_idx], nu_dn, sys_dn, e_q, l=l)
            dobs_dp.append(((w_up - w_dn) / (2 * eps)) * params_err[p_idx])
        elif obs_type == "two_photon":
            w_up = get_two_photon_width(m_up, evec_up[:, state_idx], nu_up, sys_up, e_q, l=l)
            w_dn = get_two_photon_width(m_dn, evec_dn[:, state_idx], nu_dn, sys_dn, e_q, l=l)
            dobs_dp.append(((w_up - w_dn) / (2 * eps)) * params_err[p_idx])
        elif obs_type == "rms_radius":
            r_up = calc_rms_radius(evec_up[:, state_idx], nu_up, l=l)
            r_dn = calc_rms_radius(evec_dn[:, state_idx], nu_dn, l=l)
            dobs_dp.append(((r_up - r_dn) / (2 * eps)) * params_err[p_idx])
        elif obs_type == "two_photon_pwave":
            w_up = get_two_photon_width_pwave(m_up, evec_up[:, state_idx], nu_up, sys_up, e_q, j=j)
            w_dn = get_two_photon_width_pwave(m_dn, evec_dn[:, state_idx], nu_dn, sys_dn, e_q, j=j)
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
):
    eps = 1e-4
    dobs_dp = []

    l_i, spin_i, idx_i, j_i = parse_state_name(state_i_name)
    if decay_type in ["M1", "E1"]:
        l_f, spin_f, idx_f, j_f = parse_state_name(state_f_name)

    for p_idx in range(4):
        if params_err[p_idx] == 0.0:
            dobs_dp.append(0.0)
            continue

        p_up = [sys_obj.alpha_s, sys_obj.b, sys_obj.c, sys_obj.sigma_smear]
        p_dn = [sys_obj.alpha_s, sys_obj.b, sys_obj.c, sys_obj.sigma_smear]

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
        dobs_dp.append(((w_up - w_dn) / (2 * eps)) * params_err[p_idx])

    return np.sqrt(np.sum(np.array(dobs_dp) ** 2))


def solve_sd_mixed(sys_obj, r, n_d, e_q):
    """Tensor-mix the (n_d+1)^3S_1 and (n_d)^3D_1 vectors; return the physical
    (S-like, D-like) masses and e+e- widths.

    The ^3S_1 and ^3D_1 channels share J^PC = 1^-- and are coupled by the
    off-diagonal tensor element <^3S_1|V_T|^3D_1>. Diagonalizing the 2x2
    ``[[m_S, H_SD], [H_SD, m_D]]`` gives the physical eigenstates: the lower
    eigenvalue is S-dominant (m_S < m_D throughout this radial region), the upper
    is D-dominant. The D-like state's leptonic width is carried almost entirely by
    its small ^3S_1 admixture -- a pure D-wave has R(0)=0 -- which is the whole
    reason psi(3770)/psi(4160) have measurable e+e- widths at all.

    Returns ``(m_Slike, m_Dlike, w_Slike_keV, w_Dlike_keV)`` or ``None`` if either
    radial level is absent from the basis. Widths are 0 when ``e_q == 0`` (B_c).
    """
    n_s = n_d + 1
    evS, _, ecS, nuS = solve_gem(sys_obj, r, l=0, spin=1)
    evD, _, ecD, nuD = solve_gem(sys_obj, r, l=2, spin=1)
    if n_s >= ecS.shape[1] or n_d >= ecD.shape[1]:
        return None
    m_s = get_mass(evS, ecS, nuS, sys_obj, n_s, spin=1, l=0, j=1)
    m_d = get_mass(evD, ecD, nuD, sys_obj, n_d, spin=1, l=2, j=1)
    h_sd = calc_tensor_mixing_exact(ecS[:, n_s], nuS, ecD[:, n_d], nuD, sys_obj, s=1, j=1)
    eigvals, eigvecs = np.linalg.eigh(np.array([[m_s, h_sd], [h_sd, m_d]]))
    if e_q == 0.0:
        return eigvals[0], eigvals[1], 0.0, 0.0
    w_s = get_leptonic_width_mixed(
        eigvals[0], ecS[:, n_s], nuS, ecD[:, n_d], nuD,
        eigvecs[0, 0], eigvecs[1, 0], sys_obj, e_q,
    )
    w_d = get_leptonic_width_mixed(
        eigvals[1], ecS[:, n_s], nuS, ecD[:, n_d], nuD,
        eigvecs[0, 1], eigvecs[1, 1], sys_obj, e_q,
    )
    return eigvals[0], eigvals[1], w_s, w_d


def propagate_mixed_width_uncertainty(sys_obj, r, params_err, n_d, e_q, which):
    """sigma_comp on a mixed S-D leptonic width, propagated through the *full*
    diagonalization (central differences in the 4 Cornell parameters).

    This is the honest error bar for the mixed observable: re-mixing at each
    perturbed parameter set captures how the e+e- strength tracks the S-D mixing
    angle and the S-wave |R(0)|^2. (The old code reused the bare pure-D-wave
    sigma_comp here, an underestimate that inflated the psi(3770) pull.) ``which``
    is 'S' or 'D' for the S-like / D-like physical state.
    """
    eps = 1e-4
    idx = 2 if which == "S" else 3  # position of the chosen width in solve_sd_mixed
    contribs = []
    for p_idx in range(4):
        if params_err[p_idx] == 0.0:
            contribs.append(0.0)
            continue
        p_up = [sys_obj.alpha_s, sys_obj.b, sys_obj.c, sys_obj.sigma_smear]
        p_dn = list(p_up)
        p_up[p_idx] += eps
        p_dn[p_idx] -= eps
        r_up = solve_sd_mixed(QuarkoniumSystem(sys_obj.m_1, sys_obj.m_2, *p_up), r, n_d, e_q)
        r_dn = solve_sd_mixed(QuarkoniumSystem(sys_obj.m_1, sys_obj.m_2, *p_dn), r, n_d, e_q)
        if r_up is None or r_dn is None:
            contribs.append(0.0)
            continue
        contribs.append((r_up[idx] - r_dn[idx]) / (2 * eps) * params_err[p_idx])
    return float(np.sqrt(np.sum(np.square(contribs))))


def generate_spectrum(
    sys_obj: QuarkoniumSystem,
    r,
    pdg_data,
    sector_name,
    sector_id,
    particle_names,
    params_err=None,
    pdg_err=None,
    max_n=3,
    max_l=2,
    n_fit_params=4,
    pdg_widths_ee=None,
    pdg_widths_ee_err=None,
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
                export_gem_parameters(nu_array, evecs, l_chars[l], sector_id)
                if l == 0:
                    virial_ratio = (
                        check_virial_theorem(
                            evecs[:, 0], nu_array, sys_obj, l=0, spin=spin
                        )
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
                    rms_err = 0.0
                    if params_err is not None and sum(params_err) > 0:
                        _, rms_err = propagate_uncertainty(
                            sys_obj, r, params_err, state_idx, spin, l, j, e_q,
                            "rms_radius",
                        )
                    calculated_observables[name + "_rms"] = (
                        "RMS Radius (GeV^-1)",
                        rms_rad,
                        rms_err,
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
                        # P-wave chi_J -> gamma gamma (J=0 and J=2 only; J=1 forbidden by Yang)
                        if spin == 1 and l == 1 and j in [0, 2]:
                            width_gg_p = get_two_photon_width_pwave(
                                mass,
                                evecs[:, state_idx],
                                nu_array,
                                sys_obj,
                                e_q,
                                j=j,
                            )
                            gg_p_err = 0.0
                            if params_err is not None and sum(params_err) > 0:
                                _, gg_p_err = propagate_uncertainty(
                                    sys_obj, r, params_err, state_idx, spin, l, j,
                                    e_q, "two_photon_pwave",
                                )
                            calculated_observables[name + "_gg"] = (
                                "Two-Photon Width (γγ)",
                                width_gg_p,
                                gg_p_err,
                            )

    # Tensor S-D mixing of the J^PC = 1^-- pairs: (2^3S_1, 1^3D_1) and
    # (3^3S_1, 2^3D_1). The off-diagonal <^3S_1|V_T|^3D_1> element redistributes
    # both the masses and -- crucially -- the e+e- strength: a pure D-wave has
    # R(0)=0, so psi(3770)/psi(4160) get their leptonic widths only through the
    # small S-wave admixture. Only cc and bb carry vector (e+e-) data; the B_c is
    # charged (no e+e- annihilation) and its observed states are the singlets, so
    # it is excluded. (Previously only the first pair was mixed, leaving 2^3D_1 as
    # a bare R(0)=0 width, and the mixed sigma_comp reused the pure-D-wave value.)
    sym_3 = "ψ" if "c_cbar" in sector_name else ("Υ" if "b_bbar" in sector_name else None)
    if sym_3 is not None:
        sd_pairs = [
            (0, f"(2^3S) {sym_3}", f"(1^3D_1) {sym_3}"),
            (1, f"(3^3S) {sym_3}", f"(2^3D_1) {sym_3}"),
        ]
        for n_d, s_name, d_name in sd_pairs:
            if s_name not in calculated_masses or d_name not in calculated_masses:
                continue
            mixed = solve_sd_mixed(sys_obj, r, n_d, e_q)
            if mixed is None:
                continue
            m_s, m_d, w_s, w_d = mixed
            # Mixing shifts the masses by < a few MeV, far below sigma_comp
            # (~20-40 MeV), so the pre-mixing mass uncertainty is retained.
            calculated_masses[s_name] = (m_s, calculated_masses[s_name][1])
            calculated_masses[d_name] = (m_d, calculated_masses[d_name][1])
            if e_q != 0.0:
                ws_err = wd_err = 0.0
                if params_err is not None and sum(params_err) > 0:
                    ws_err = propagate_mixed_width_uncertainty(
                        sys_obj, r, params_err, n_d, e_q, "S"
                    )
                    wd_err = propagate_mixed_width_uncertainty(
                        sys_obj, r, params_err, n_d, e_q, "D"
                    )
                calculated_observables[s_name] = ("Leptonic Width (e+e-)", w_s, ws_err)
                calculated_observables[d_name] = ("Leptonic Width (e+e-)", w_d, wd_err)

    # Derived per-state theory sigma (MeV) = magnitude of the leading omitted
    # relativistic correction for each state, from the same eigenvectors the
    # masses came from. This is an informational model-viability diagnostic
    # (recorded as the Sigma_Theory_MeV column / figure band) -- it is NOT the
    # pull denominator. The pull/chi-square divides by the *computational* sigma:
    # quad(sigma_exp, mass_err), where mass_err is the propagated parameter
    # covariance computed above. (The fit weight in fitter.py still uses this
    # theory sigma as a well-posed regularizer; the validation metric does not.)
    sigma_theory_mev = {}
    for name, evec in calculated_evecs.items():
        if name not in calculated_masses:
            continue
        l_s, _, _, _ = parse_state_name(name)
        dE_rel, _ = calc_relativistic_shift(evec, nu_array, sys_obj, l_s)
        sigma_theory_mev[name] = dE_rel * 1000.0

    # S-wave vector (^3S_1) leptonic widths to fold into the combined validation
    # chi-square. Matched to experiment by the bare state label; only states whose
    # label ends "^3S)" are included, so the D-wave e+e- widths are excluded (they
    # are model-completeness cases, not implementation tests -- see metrics).
    leptonic_obs = []
    if pdg_widths_ee:
        for nm, (otype, val, err) in calculated_observables.items():
            if "Leptonic" not in otype:
                continue
            bare = nm.split()[0]
            if not bare.endswith("^3S)"):
                continue
            g_exp = pdg_widths_ee.get(bare)
            if g_exp is None:
                continue
            s_exp = (pdg_widths_ee_err or {}).get(bare, 0.0)
            leptonic_obs.append((nm, val, err, g_exp, s_exp))

    gof = format_and_evaluate(
        calculated_masses, pdg_data, sector_name, sector_id,
        pdg_err=pdg_err, n_fit_params=n_fit_params,
        sigma_theory_mev=sigma_theory_mev, leptonic_obs=leptonic_obs,
    )

    if calculated_observables:
        print(f"\n--- Decay Observables for {sector_name} ---")
        for state, (obs_type, obs_val, obs_err) in calculated_observables.items():
            print(f"{state:<15} | {obs_type}: {obs_val:.2f} ± {obs_err:.2f} keV")

        export_observables(calculated_observables, sector_id)

    return calculated_masses, calculated_evecs, nu_array, gof


if __name__ == "__main__":
    # Experimental data is read live from the PDG SQLite dump (data/pdg-*.sqlite).
    all_pdg = load_pdg_data()

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

    # Reduced masses for interpolation (defined before pre-fit calls)
    mu_cc = 1.500 / 2.0
    mu_bb = 4.730 / 2.0
    mu_bc = (4.730 * 1.500) / (4.730 + 1.500)

    # Electric charge per sector (for the leptonic-width fit residuals).
    e_q_map = {"cc": 2.0 / 3.0, "bb": -1.0 / 3.0, "bc": 0.0}

    # We need to get bb and cc alpha_s and sigma first to do the B_c interpolation
    (cc_alpha_s, _, _, cc_sigma), cc_errs = get_or_fit_parameters(
        1.500,
        1.500,
        all_pdg.get("cc", {}),
        r,
        [0.400, 0.183, -0.250, 1.09],
        paths.params_csv("cc"),
        pdg_mass_err=all_pdg.get("cc_mass_err_GeV", {}),
        decay_targets=all_pdg.get("cc_widths_ee_keV", {}),
        e_q=e_q_map["cc"],
    )
    (bb_alpha_s, _, _, bb_sigma), bb_errs = get_or_fit_parameters(
        4.730,
        4.730,
        all_pdg.get("bb", {}),
        r,
        [0.350, 0.193, 0.030, 1.34],
        paths.params_csv("bb"),
        pdg_mass_err=all_pdg.get("bb_mass_err_GeV", {}),
        decay_targets=all_pdg.get("bb_widths_ee_keV", {}),
        e_q=e_q_map["bb"],
    )

    # QFT Logarithmic Interpolation of alpha_s and linear interpolation of sigma
    x_bc = np.log(mu_bc / mu_cc) / np.log(mu_bb / mu_cc)
    inv_alpha_cc = 1.0 / cc_alpha_s
    inv_alpha_bb = 1.0 / bb_alpha_s
    inv_alpha_bc = inv_alpha_cc + x_bc * (inv_alpha_bb - inv_alpha_cc)
    bc_alpha_s_qft = 1.0 / inv_alpha_bc
    bc_sigma_qft = cc_sigma + x_bc * (bb_sigma - cc_sigma)

    # --- Propagate the cc/bb fit uncertainties into the interpolated B_c inputs.
    # alpha_s is NOT fitted freely for B_c (it is fixed by the log-interpolation
    # of 1/alpha_s in the reduced-mass scale), so its uncertainty is inherited
    # entirely from the charm and bottom sectors. Likewise for the smearing sigma.
    err_cc_alpha, err_bb_alpha = cc_errs[0], bb_errs[0]
    err_cc_sigma, err_bb_sigma = cc_errs[3], bb_errs[3]

    # d(alpha_bc)/d(alpha_cc) and d(alpha_bc)/d(alpha_bb) from
    # 1/alpha_bc = (1-x)/alpha_cc + x/alpha_bb
    d_bc_d_cc = (1.0 - x_bc) / (inv_alpha_bc**2 * cc_alpha_s**2)
    d_bc_d_bb = x_bc / (inv_alpha_bc**2 * bb_alpha_s**2)
    err_bc_alpha = np.hypot(d_bc_d_cc * err_cc_alpha, d_bc_d_bb * err_bb_alpha)
    # sigma_bc = (1-x) sigma_cc + x sigma_bb  (linear interpolation)
    err_bc_sigma = np.hypot((1.0 - x_bc) * err_cc_sigma, x_bc * err_bb_sigma)
    # [err_alpha_s, err_b, err_c, err_sigma]; b and c are exactly determined by
    # the 2 B_c data points (dof = 0), so their fit covariance is unavailable and
    # the interpolated coupling dominates the predicted B_c uncertainties.
    bc_param_errs = [err_bc_alpha, 0.0, 0.0, err_bc_sigma]
    print(
        f"\nInterpolated B_c inputs: alpha_s = {bc_alpha_s_qft:.4f} ± {err_bc_alpha:.4f}, "
        f"sigma = {bc_sigma_qft:.4f} ± {err_bc_sigma:.4f}  (inherited from cc/bb fits)"
    )

    sectors_config = [
        {
            "id": "bb",
            "name": "Bottomonium (b_bbar)",
            "m_1": 4.730,
            "m_2": 4.730,
            "pdg_data": all_pdg.get("bb", {}),
            "names": bb_names,
            "initial_guesses": [0.350, 0.193, 0.030, 1.34],
            "bounds": None,
            "max_n": 3,
            "max_l": 2,
            "n_fit_params": 4,  # alpha_s, b, c, sigma all free
        },
        {
            "id": "cc",
            "name": "Charmonium (c_cbar)",
            "m_1": 1.500,
            "m_2": 1.500,
            "pdg_data": all_pdg.get("cc", {}),
            "names": cc_names,
            "initial_guesses": [0.400, 0.183, -0.250, 1.09],
            "bounds": None,
            "max_n": 3,
            "max_l": 2,
            "n_fit_params": 4,  # alpha_s, b, c, sigma all free
        },
        {
            "id": "bc",
            "name": "B_c Meson (b_cbar)",
            "m_1": 4.730,
            "m_2": 1.500,
            "pdg_data": all_pdg.get("bc", {}),
            "names": bc_names,
            "initial_guesses": [bc_alpha_s_qft, 0.183, -0.090, bc_sigma_qft],
            "bounds": (
                [bc_alpha_s_qft - 1e-5, 0.1, -1.0, bc_sigma_qft - 1e-5],
                [bc_alpha_s_qft + 1e-5, 0.35, 1.0, bc_sigma_qft + 1e-5],
            ),
            "max_n": 3,
            "max_l": 2,
            # alpha_s and sigma are frozen to the interpolated values, so only
            # b and c are genuinely free degrees of freedom for the B_c fit.
            "n_fit_params": 2,
            "override_errs": bc_param_errs,
        },
    ]

    results_dict = {}
    gof_summary = []

    for config in sectors_config:
        print(f"\n--- Fitting/Loading {config['name']} Parameters ---")
        sid = config["id"]
        pdg_mass_err = all_pdg.get(f"{sid}_mass_err_GeV", {})
        (alpha_s, b, c, sigma), errs = get_or_fit_parameters(
            m_1=config["m_1"],
            m_2=config["m_2"],
            pdg_data=config["pdg_data"],
            r=r,
            initial_guesses=config["initial_guesses"],
            csv_path=paths.params_csv(sid),
            bounds=config["bounds"],
            pdg_mass_err=pdg_mass_err,
            decay_targets=all_pdg.get(f"{sid}_widths_ee_keV", {}),
            e_q=e_q_map.get(sid, 0.0),
        )
        # For sectors with frozen/under-determined parameters, substitute the
        # externally propagated uncertainties (e.g. B_c inherits the interpolated
        # alpha_s/sigma errors from the cc and bb fits).
        if config.get("override_errs") is not None:
            errs = list(config["override_errs"])
        sys_obj = QuarkoniumSystem(
            m_1=config["m_1"], m_2=config["m_2"], alpha_s=alpha_s, b=b, c=c,
            sigma_smear=sigma,
        )
        masses, evecs, nu, gof = generate_spectrum(
            sys_obj,
            r,
            config["pdg_data"],
            config["name"],
            sid,
            config["names"],
            params_err=errs,
            pdg_err=pdg_mass_err,
            max_n=config["max_n"],
            max_l=config["max_l"],
            n_fit_params=config["n_fit_params"],
            pdg_widths_ee=all_pdg.get(f"{sid}_widths_ee_keV", {}),
            pdg_widths_ee_err=all_pdg.get(f"{sid}_widths_ee_err_keV", {}),
        )
        gof["sector_id"] = config["id"]
        gof_summary.append(gof)
        results_dict[config["id"]] = {
            "masses": masses,
            "evecs": evecs,
            "nu": nu,
            "sys_obj": sys_obj,
            "errs": errs,
        }

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
            [
                ("(1^3P_0) χ_c", "(1^3S) J/ψ"),
                ("(1^3P_1) χ_c", "(1^3S) J/ψ"),
                ("(1^3P_2) χ_c", "(1^3S) J/ψ"),
            ],
        ),
        (
            "bb",
            -1.0 / 3.0,
            [("(1^3S) Υ_b", "(1^1S) η_b"), ("(2^3S) Υ", "(2^1S) η_b")],
            [
                ("(1^3P_0) χ_b", "(1^3S) Υ_b"),
                ("(1^3P_1) χ_b", "(1^3S) Υ_b"),
                ("(1^3P_2) χ_b", "(1^3S) Υ_b"),
            ],
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
        rad_csv_path = paths.summary_csv("radiative_decays.csv")
        rad_df.to_csv(rad_csv_path, index=False)
        print(f"\nExported radiative transitions to {rad_csv_path}")

    print("=" * 80 + "\n")

    # =========================================================================
    # GLOBAL GOODNESS-OF-FIT SUMMARY (chi^2 per sector)
    # =========================================================================
    print("=" * 80)
    print("--- Global Goodness-of-Fit Summary (mass spectrum) ---")
    print(
        f"{'Sector':<22} | {'N':<4} | {'chi^2':<10} | {'dof':<5} | {'chi^2/dof':<10} | {'RMS [MeV]':<10}"
    )
    print("-" * 80)
    total_chi2 = 0.0
    total_n = 0
    total_par = 0
    for gof in gof_summary:
        total_chi2 += gof["chi2"]
        total_n += gof["n"]
        total_par += gof["n"] - gof["dof"]
        print(
            f"{gof['sector']:<22} | {gof['n']:<4} | {gof['chi2']:<10.2f} | "
            f"{gof['dof']:<5} | {gof['chi2_per_dof']:<10.3f} | {gof['rms_mev']:<10.2f}"
        )
    global_dof = max(total_n - total_par, 1)
    print("-" * 80)
    print(
        f"{'GLOBAL':<22} | {total_n:<4} | {total_chi2:<10.2f} | "
        f"{total_n - total_par:<5} | {total_chi2 / global_dof:<10.3f} |"
    )
    print("=" * 80 + "\n")

    gof_df = pd.DataFrame(gof_summary)[
        ["sector_id", "sector", "n", "chi2", "dof", "chi2_per_dof", "rms_mev",
         "n_lep", "chi2_comb", "dof_comb", "chi2_per_dof_comb"]
    ]
    gof_csv_path = paths.summary_csv("goodness_of_fit.csv")
    gof_df.to_csv(gof_csv_path, index=False)
    print(f"Goodness-of-fit summary saved to {gof_csv_path}\n")

    generate_consolidated_report()
