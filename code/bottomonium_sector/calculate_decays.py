#!/usr/bin/env python3

import numpy as np
from scipy.integrate import simpson
from bottomonium_sector import run_comparisons, ALPHA_S

# --- Physical Constants ---
ALPHA_EM = 1.0 / 137.036


def calc_leptonic_width(u_array, r_array, m_v, e_q=-1 / 3):
    """
    Calculates the leptonic decay width V -> e+ e- using the Van Royen-Weisskopf formula
    for Vector (S=1) states.
    """
    du_dr = np.gradient(u_array, r_array)
    psi_0_sq = (du_dr[0] ** 2) / (4.0 * np.pi)
    gamma_gev = (16.0 * np.pi * ALPHA_EM**2 * (e_q**2) * psi_0_sq) / (m_v**2)
    return gamma_gev * 1e6  # Convert GeV to keV


def calc_two_photon_width(u_array, r_array, m_p, e_q=-1 / 3):
    """
    Calculates the two-photon decay width P -> gamma gamma for Pseudoscalar (S=0) states.
    Note the dependence on e_q^4 and the factor of 12 instead of 16.
    """
    du_dr = np.gradient(u_array, r_array)
    psi_0_sq = (du_dr[0] ** 2) / (4.0 * np.pi)
    gamma_gev = (12.0 * np.pi * ALPHA_EM**2 * (e_q**4) * psi_0_sq) / (m_p**2)
    return gamma_gev * 1e6  # Convert GeV to keV


def calc_e1_transition_width(
    u_initial, u_final, r_array, m_initial, m_final, e_q=-1 / 3, C_fi=1.0
):
    """
    Calculates the Electric Dipole (E1) radiative transition width.
    """
    omega = m_initial - m_final
    if omega <= 0:
        return 0.0

    integrand = u_final * r_array * u_initial
    dipole_matrix_element = simpson(y=integrand, x=r_array)

    gamma_gev = (
        (4.0 / 3.0)
        * ALPHA_EM
        * (e_q**2)
        * (omega**3)
        * (dipole_matrix_element**2)
        * C_fi
    )
    return gamma_gev * 1e6  # Convert GeV to keV


def calculate_bottomonium_decays():
    print("Gathering GEM wavefunctions and masses for Bottomonium...")
    results = run_comparisons()

    r = results["r"]

    # 1. Extract the GEM wavefunctions
    u1s = results["u1s_gem"]
    u2s = results["u2s_gem"]
    u1p = results["u_gem_1"][:, 0]

    # 2. Extract calculated masses
    mass_dict = {row[0]: row[1] for row in results["comparison_table_data"]}

    # S-Wave States
    m_eta_b_1s = mass_dict.get("(1^1S) η_b")
    m_ups_1s = mass_dict.get("(1^3S) Υ_b")
    m_ups_2s = mass_dict.get("(2^3S) Υ")

    # P-Wave States
    m_chi_b0 = mass_dict.get("(1^3P_0) χ_b0")
    m_chi_b1 = mass_dict.get("(1^3P_1) χ_b1")
    m_chi_b2 = mass_dict.get("(1^3P_2) χ_b2")

    print("\n" + "=" * 65)
    print(" COMPREHENSIVE DECAY WIDTH ANALYSIS (BOTTOMONIUM) ".center(65))
    print("=" * 65)

    # --- S-WAVE ANNIHILATION DECAYS ---
    print("\n--- 1. Annihilation Decays (VRW & Two-Photon) ---")
    qcd_corr_lep = 1.0 - (16.0 / (3.0 * np.pi)) * ALPHA_S
    qcd_corr_gg = 1.0 - (3.4 / np.pi) * ALPHA_S  # Approx QCD correction for 2-photon

    if m_ups_1s and m_ups_2s:
        w_ups1s = calc_leptonic_width(u1s, r, m_ups_1s)
        w_ups2s = calc_leptonic_width(u2s, r, m_ups_2s)
        print(
            f"[*] Υ(1S) -> e+ e-  | Raw: {w_ups1s:>6.3f} keV | QCD Corr: {w_ups1s * qcd_corr_lep:>5.3f} keV"
        )
        print(
            f"[*] Υ(2S) -> e+ e-  | Raw: {w_ups2s:>6.3f} keV | QCD Corr: {w_ups2s * qcd_corr_lep:>5.3f} keV"
        )
        print(
            f"    -> Leptonic Ratio Γ(2S)/Γ(1S) : {w_ups2s / w_ups1s:.3f} (Exp: ~0.455)"
        )

    if m_eta_b_1s:
        w_eta_1s = calc_two_photon_width(
            u1s, r, m_eta_b_1s
        )  # Uses same spatial u(r) as Upsilon
        print(
            f"[*] η_b(1S) -> γγ   | Raw: {w_eta_1s:>6.3f} keV | QCD Corr: {w_eta_1s * qcd_corr_gg:>5.3f} keV"
        )

    # --- P-WAVE RADIATIVE TRANSITIONS ---
    print("\n--- 2. Electric Dipole (E1) Radiative Transitions ---")

    # Experimental target masses for precise phase space correction
    exp_ups1s = 9.460
    exp_chi_b = {
        "χ_b0": {"calc": m_chi_b0, "exp": 9.859},
        "χ_b1": {"calc": m_chi_b1, "exp": 9.892},
        "χ_b2": {"calc": m_chi_b2, "exp": 9.912},
    }

    for name, data in exp_chi_b.items():
        if data["calc"] and m_ups_1s:
            w_raw = calc_e1_transition_width(
                u_initial=u1p,
                u_final=u1s,
                r_array=r,
                m_initial=data["calc"],
                m_final=m_ups_1s,
                C_fi=1.0 / 3.0,
            )

            # Phase space correction (omega_exp^3 / omega_calc^3)
            omega_calc = data["calc"] - m_ups_1s
            omega_exp = data["exp"] - exp_ups1s
            w_corr = w_raw * ((omega_exp / omega_calc) ** 3)

            print(
                f"[*] {name}(1P) -> Υ(1S) + γ | Raw Phase Space: {w_raw:>6.3f} keV | Exp Phase Space: {w_corr:>6.3f} keV"
            )

    print("=" * 65 + "\n")


if __name__ == "__main__":
    calculate_bottomonium_decays()
