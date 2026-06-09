import numpy as np
from scipy.optimize import least_squares
from .gem_solver import analytical_integral

# Fundamental QFT Constants
ALPHA_EM = 1.0 / 137.036


def get_running_alpha_s(sys):
    """
    Returns the appropriate high-energy running coupling constant for annihilation decays.
    """
    if sys.m_1 > 4.0:
        # Bottomonium (m_b ≈ 4.73 GeV)
        alpha_s_run = 0.20
    else:
        # Charmonium (m_c ≈ 1.5 GeV)
        alpha_s_run = 0.35
    return alpha_s_run


def calc_R_origin_sq_hypervirial(c_vec, nu_array, sys, l=0):
    """
    Calculates |R(0)|^2 natively using the Schwinger/Hypervirial Theorem.
    R(0)^2 = 2 * mu * < dV/dr >
    """
    if l > 0:
        return 0.0

    # Reconstruct the normalized c_i coefficients for the u(r) basis
    norms = np.sqrt(analytical_integral(2, 2.0 * nu_array))
    c_norm = c_vec / norms

    # Evaluate the exact expectation matrix for <1/r^2>
    nu_ij = nu_array[:, np.newaxis] + nu_array[np.newaxis, :]
    I_0 = analytical_integral(0, nu_ij)

    exp_1_over_r2 = np.sum(c_norm[:, np.newaxis] * c_norm[np.newaxis, :] * I_0)

    # R(0)^2 = 2 * mu * [ (4*alpha_s / 3) * <1/r^2> + b ]
    r_origin_sq = 2.0 * sys.mu * ((4.0 * sys.alpha_s / 3.0) * exp_1_over_r2 + sys.b)

    return r_origin_sq


def calc_R_second_deriv_origin_sq(c_vec, nu_array, l=2):
    """
    Calculates |R''(0)|^2 analytically for D-wave (L=2) states.
    For a Gaussian basis R(r) = sum c_i * r^2 * exp(-nu_i * r^2), R''(0) = 2 * sum(c_i).
    """
    if l != 2:
        return 0.0

    norms = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array))
    c_norm = c_vec / norms

    r_double_prime_0 = 2.0 * np.sum(c_norm)
    return r_double_prime_0**2


def get_leptonic_width_mixed(
    mass_GeV, c_vec_S, nu_array_S, c_vec_D, nu_array_D, mix_S, mix_D, sys, e_q
):
    """
    Calculates the leptonic decay width (V -> e+ e-) for a mixed S-D state.
    """
    alpha_s_run = get_running_alpha_s(sys)
    qcd_correction = 1.0 - (16.0 * alpha_s_run) / (3.0 * np.pi)

    # Evaluate magnitudes and dynamically retrieve the parity phase/sign
    R_0_sq = calc_R_origin_sq_hypervirial(c_vec_S, nu_array_S, sys, l=0)
    norms_S = np.sqrt(analytical_integral(2, 2.0 * nu_array_S))
    sign_S = np.sign(np.sum(c_vec_S / norms_S))
    R_0 = sign_S * np.sqrt(R_0_sq)

    R_double_prime_0_sq = calc_R_second_deriv_origin_sq(c_vec_D, nu_array_D, l=2)
    norms_D = np.sqrt(analytical_integral(6, 2.0 * nu_array_D))
    sign_D = np.sign(np.sum(c_vec_D / norms_D))
    R_double_prime_0 = sign_D * np.sqrt(R_double_prime_0_sq)

    # Create the component transition amplitudes
    amp_S = (2.0 * ALPHA_EM * np.abs(e_q) / mass_GeV) * R_0
    amp_D = (
        5.0 * ALPHA_EM * np.abs(e_q) / (np.sqrt(2.0) * sys.m_1**2 * mass_GeV)
    ) * R_double_prime_0

    total_amp = mix_S * amp_S + mix_D * amp_D
    width_GeV = (total_amp**2) * qcd_correction

    return width_GeV * 1e6


def get_leptonic_width(mass_GeV, c_vec, nu_array, sys, e_q, l=0):
    """
    Calculates the leptonic decay width (V -> e+ e-) for vector mesons (Spin=1).
    Supports S-wave (L=0) and D-wave (L=2) annihilations.
    """
    alpha_s_run = get_running_alpha_s(sys)
    # First-order perturbative QCD correction using the running coupling
    qcd_correction = 1.0 - (16.0 * alpha_s_run) / (3.0 * np.pi)

    if l == 0:
        R_0_sq = calc_R_origin_sq_hypervirial(c_vec, nu_array, sys, l=l)
        width_GeV = (
            (4.0 * ALPHA_EM**2 * e_q**2) / (mass_GeV**2) * R_0_sq * qcd_correction
        )
    elif l == 2:
        R_double_prime_0_sq = calc_R_second_deriv_origin_sq(c_vec, nu_array, l=l)
        width_GeV = (
            (25.0 * ALPHA_EM**2 * e_q**2)
            / (2.0 * sys.m_1**4 * mass_GeV**2)
            * R_double_prime_0_sq
            * qcd_correction
        )
    else:
        return 0.0

    return width_GeV * 1e6  # Return in keV


def get_two_photon_width(mass_GeV, c_vec, nu_array, sys, e_q, l=0):
    """
    Calculates the two-photon decay width (P -> gamma gamma) for pseudoscalar mesons (Spin=0).
    """
    R_0_sq = calc_R_origin_sq_hypervirial(c_vec, nu_array, sys, l=l)

    alpha_s_run = get_running_alpha_s(sys)
    qcd_correction = 1.0 - (3.4 * alpha_s_run) / np.pi

    width_GeV = (12.0 * ALPHA_EM**2 * e_q**4) / (mass_GeV**2) * R_0_sq * qcd_correction
    return width_GeV * 1e6  # Return in keV


def calc_overlap_3p0_simplified(c_A, nu_A, c_B, nu_B, c_C, nu_C, l_A=0, l_B=0, l_C=0):
    """
    Calculates the phenomenological spatial overlap integral for a 3P0 hadronic decay A -> B + C.
    """
    overlap = 0.0
    norms_A = np.sqrt(analytical_integral(2 * l_A + 2, 2.0 * nu_A))
    norms_B = np.sqrt(analytical_integral(2 * l_B + 2, 2.0 * nu_B))
    norms_C = np.sqrt(analytical_integral(2 * l_C + 2, 2.0 * nu_C))

    c_norm_A = c_A / norms_A
    c_norm_B = c_B / norms_B
    c_norm_C = c_C / norms_C

    for i in range(len(nu_A)):
        for j in range(len(nu_B)):
            for k in range(len(nu_C)):
                nu_sum = nu_A[i] + nu_B[j] + nu_C[k]
                p_sum = l_A + l_B + l_C + 2
                I_ijk = analytical_integral(p_sum, nu_sum)
                overlap += c_norm_A[i] * c_norm_B[j] * c_norm_C[k] * I_ijk
    return overlap


def get_3p0_decay_width(
    mass_A, mass_B, mass_C, c_A, nu_A, c_B, nu_B, c_C, nu_C, l_A=0, gamma_3p0=0.4
):
    if mass_A < mass_B + mass_C:
        return 0.0

    # Fully relativistic momentum phase space factor
    # Matches p = sqrt(M_initial^2 / 4 - M_final^2) if M_B == M_C
    if np.isclose(mass_B, mass_C):
        p = np.sqrt(mass_A**2 / 4.0 - mass_B**2)
    else:
        p = np.sqrt(
            (mass_A**2 - (mass_B + mass_C) ** 2) * (mass_A**2 - (mass_B - mass_C) ** 2)
        ) / (2.0 * mass_A)
    overlap = calc_overlap_3p0_simplified(
        c_A, nu_A, c_B, nu_B, c_C, nu_C, l_A=l_A, l_B=0, l_C=0
    )
    width = (
        2.0
        * np.pi
        * p
        * (np.sqrt(mass_B**2 + p**2) * np.sqrt(mass_C**2 + p**2) / mass_A)
        * (gamma_3p0**2)
        * (overlap**2)
    )
    return width * 1000.0  # Return in MeV


def tune_gamma_3p0(
    target_width_MeV,
    mass_A,
    mass_B,
    mass_C,
    c_A,
    nu_A,
    c_B,
    nu_B,
    c_C,
    nu_C,
    l_A=0,
    initial_gamma=0.4,
):
    """
    Optimizer script/function that adjusts the dimensionless vacuum pair-creation
    parameter gamma until the calculated width matches the experimental target.
    """

    def objective(gamma):
        calc_width = get_3p0_decay_width(
            mass_A,
            mass_B,
            mass_C,
            c_A,
            nu_A,
            c_B,
            nu_B,
            c_C,
            nu_C,
            l_A=l_A,
            gamma_3p0=gamma[0],
        )
        return calc_width - target_width_MeV

    res = least_squares(objective, [initial_gamma], bounds=([0.0], [np.inf]))
    return res.x[0]


def calc_R_prime_origin_sq(c_vec, nu_array, l=1):
    """
    Calculates |R'(0)|^2 for P-wave (L=1) states.
    For R(r) = sum c_i * r * exp(-nu_i * r^2), R'(0) = sum c_i / norm_i.
    """
    if l != 1:
        return 0.0
    norms = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array))
    c_norm = c_vec / norms
    r_prime_0 = np.sum(c_norm)
    return r_prime_0**2


def get_two_photon_width_pwave(mass_GeV, c_vec, nu_array, sys, e_q, j=0):
    """
    Calculates the two-photon decay width for P-wave triplet states (chi_cJ, chi_bJ).
    Uses the Barbieri-Gatto-Remiddi (1976) formula with |R'(0)|^2.
    Valid for j=0 (chi_0 -> gamma gamma) and j=2 (chi_2 -> gamma gamma).
    chi_1 (j=1) is forbidden by Yang's theorem.
    QCD corrections at O(alpha_s) are not applied — they diverge for P-wave scalar.
    """
    if j not in [0, 2]:
        return 0.0

    R_prime_0_sq = calc_R_prime_origin_sq(c_vec, nu_array, l=1)
    m_q = sys.m_1  # constituent quark mass

    if j == 0:
        coeff = 27.0 / 2.0   # chi_0 coefficient (BGR 1976)
    else:
        coeff = 18.0 / 5.0   # chi_2 = (4/15) * (27/2)

    width_GeV = coeff * ALPHA_EM**2 * e_q**4 * R_prime_0_sq / m_q**4
    return width_GeV * 1e6  # keV


def get_m1_decay_width(mass_i, mass_f, c_i, nu_i, c_f, nu_f, sys, e_q):
    """
    Calculates the Magnetic Dipole (M1) radiative transition width (e.g., V -> P + gamma).
    """
    if mass_i <= mass_f:
        return 0.0

    k = (mass_i**2 - mass_f**2) / (2.0 * mass_i)

    overlap = 0.0
    norms_i = np.sqrt(analytical_integral(2, 2.0 * nu_i))
    norms_f = np.sqrt(analytical_integral(2, 2.0 * nu_f))

    c_norm_i = c_i / norms_i
    c_norm_f = c_f / norms_f

    for idx_i in range(len(nu_i)):
        for idx_f in range(len(nu_f)):
            nu_sum = nu_i[idx_i] + nu_f[idx_f]
            I_ij = analytical_integral(2, nu_sum)  # L=0 overlap integral p=2
            overlap += c_norm_i[idx_i] * c_norm_f[idx_f] * I_ij

    magnetic_factor = (1.0 / sys.m_1 + 1.0 / sys.m_2) ** 2 / 4.0
    width_GeV = (
        (4.0 / 3.0) * ALPHA_EM * (e_q**2) * (k**3) * magnetic_factor * (overlap**2)
    )

    return width_GeV * 1e6  # Return in keV


def get_e1_decay_width(mass_i, mass_f, c_i, nu_i, c_f, nu_f, sys, e_q):
    """
    Calculates the Electric Dipole (E1) radiative transition width for a
    P -> S + gamma transition (e.g. chi_cJ -> J/psi gamma).

    The dipole matrix element is the radial expectation value of the position
    operator,

        <f|r|i> = int_0^inf u_f(r) * r * u_i(r) dr,

    with u(r) = r R(r). For the Gaussian basis u_i = r^(l+1) e^{-nu r^2}, the
    integrand of a P(l=1) -> S(l=0) transition is r^{l_f+1} * r * r^{l_i+1}
    = r^{0+1 + 1 + 1+1} = r^4, so the analytic moment is I_4 (NOT the bare
    overlap I_3 -- the explicit factor of r from the dipole operator must be
    kept). This matches Eq. (e1) of the accompanying paper.
    """
    if mass_i <= mass_f:
        return 0.0

    k = (mass_i**2 - mass_f**2) / (2.0 * mass_i)

    matrix_element = 0.0
    norms_i = np.sqrt(analytical_integral(4, 2.0 * nu_i))  # P-wave norm (L=1 -> p=4)
    norms_f = np.sqrt(analytical_integral(2, 2.0 * nu_f))  # S-wave norm (L=0 -> p=2)

    c_norm_i = c_i / norms_i
    c_norm_f = c_f / norms_f

    for idx_i in range(len(nu_i)):
        for idx_f in range(len(nu_f)):
            nu_sum = nu_i[idx_i] + nu_f[idx_f]
            # <f|r|i>: u_f * r * u_i = r^(L_f+1) * r * r^(L_i+1) = r^4 for P->S
            I_ij = analytical_integral(4, nu_sum)
            matrix_element += c_norm_i[idx_i] * c_norm_f[idx_f] * I_ij

    width_GeV = (4.0 / 9.0) * ALPHA_EM * (e_q**2) * (k**3) * (matrix_element**2)

    return width_GeV * 1e6  # Return in keV
