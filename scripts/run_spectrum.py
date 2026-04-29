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
)
from quarkonia.observables import get_mass, check_virial_theorem
from quarkonia.fitter import get_or_fit_parameters


def generate_spectrum(
    sys_obj: QuarkoniumSystem,
    r,
    pdg_data,
    sector_name,
    particle_names,
    max_n=3,
    max_l=2,
):
    """
    Dynamically generates the full spectroscopic multiplet n^(2S+1)L_J
    and maps them to standard meson names.
    """
    calculated_masses = {}
    calculated_wavefuncs = {}

    l_chars = {0: "S", 1: "P", 2: "D", 3: "F"}

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
                    calculated_masses[name] = mass
                    calculated_wavefuncs[name] = u_gem[:, state_idx]

    format_and_evaluate(calculated_masses, pdg_data, sector_name)


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
    bb_alpha_s, bb_b, bb_c = get_or_fit_parameters(
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
    generate_spectrum(
        bb_sys,
        r,
        all_pdg.get("bb", {}),
        "Bottomonium (b_bbar)",
        bb_names,
    )

    print("\n--- Fitting/Loading Charmonium Parameters ---")
    cc_alpha_s, cc_b, cc_c = get_or_fit_parameters(
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
    generate_spectrum(
        cc_sys,
        r,
        all_pdg.get("cc", {}),
        "Charmonium (c_cbar)",
        cc_names,
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

    bc_alpha_s, bc_b, bc_c = get_or_fit_parameters(
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
    generate_spectrum(
        bc_sys,
        r,
        all_pdg.get("bc", {}),
        "B_c Meson (b_cbar)",
        bc_names,
    )
