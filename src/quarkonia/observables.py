import numpy as np
from scipy.special import gamma
from scipy.integrate import simpson
from .gem_solver import analytical_integral


def calc_so_shift_exact(c_vec, nu_array, sys, l, s, j):
    if l == 0 or s == 0:
        return 0.0

    ls_dot = 0.5 * (j * (j + 1) - l * (l + 1) - s * (s + 1))
    v_ls_coulomb_coeff = (4.0 * sys.alpha_s / 3.0) * (
        1.0 / (sys.m_1 * sys.m_2) + 0.25 * (1.0 / sys.m_1**2 + 1.0 / sys.m_2**2)
    )
    v_ls_linear_coeff = -0.25 * sys.b * (1.0 / sys.m_1**2 + 1.0 / sys.m_2**2)

    shift = 0.0
    for i in range(len(nu_array)):
        norm_i = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array[i]))
        for k in range(len(nu_array)):
            norm_k = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array[k]))
            nu_ik = nu_array[i] + nu_array[k]
            term_coulomb = v_ls_coulomb_coeff * analytical_integral(2 * l - 1, nu_ik)
            term_linear = v_ls_linear_coeff * analytical_integral(2 * l + 1, nu_ik)
            c_i = c_vec[i] / norm_i
            c_k = c_vec[k] / norm_k
            shift += c_i * c_k * (term_coulomb + term_linear)

    return shift * ls_dot


def calc_tensor_shift_exact(c_vec, nu_array, sys, l, s, j):
    if l == 0 or s == 0 or s != 1:
        return 0.0

    if j == l - 1:
        s12 = -(l + 1.0) / (2.0 * l - 1.0)
    elif j == l:
        s12 = 1.0
    elif j == l + 1:
        s12 = -l / (2.0 * l + 3.0)
    else:
        s12 = 0.0

    v_tensor_coeff = (4.0 * sys.alpha_s) / (3.0 * sys.m_1 * sys.m_2)

    shift = 0.0
    for i in range(len(nu_array)):
        norm_i = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array[i]))
        for k in range(len(nu_array)):
            norm_k = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array[k]))
            nu_ik = nu_array[i] + nu_array[k]
            term_tensor = v_tensor_coeff * analytical_integral(2 * l - 1, nu_ik)
            c_i = c_vec[i] / norm_i
            c_k = c_vec[k] / norm_k
            shift += c_i * c_k * term_tensor

    return shift * s12


def calc_tensor_mixing_exact(c_vec_l0, nu_array_l0, c_vec_l2, nu_array_l2, sys, s, j):
    """
    Calculates the off-diagonal tensor matrix element <^3S_1 | V_T | ^3D_1>.
    This mixes states with L=0 and L=2 for S=1, J=1.

    Parameters
    ----------
    c_vec_l0 : numpy.ndarray
        Eigenvector coefficients for the L=0 state.
    nu_array_l0 : numpy.ndarray
        Gaussian basis widths for the L=0 state.
    c_vec_l2 : numpy.ndarray
        Eigenvector coefficients for the L=2 state.
    nu_array_l2 : numpy.ndarray
        Gaussian basis widths for the L=2 state.
    sys : QuarkoniumSystem
        Quarkonium system representation object.
    s : int
        Spin quantum number.
    j : int
        Total angular momentum quantum number.

    Returns
    -------
    float
        The computed tensor mixing matrix element.
    """
    if s != 1 or j != 1:  # Only for S=1, J=1 states
        return 0.0

    # S12 for mixing L=0 and L=2
    # The tensor operator couples L to L' = L +/- 2.
    # For L=0, L'=2, the S12 factor is sqrt(8)/5 (from standard Clebsch-Gordan coefficients)
    s12_mixing = np.sqrt(8.0) / 5.0  # For <L=0, S=1, J=1 | S12 | L=2, S=1, J=1>

    v_tensor_coeff = (4.0 * sys.alpha_s) / (3.0 * sys.m_1 * sys.m_2)

    mixing_element = 0.0
    for i in range(len(nu_array_l0)):
        norm_i = np.sqrt(analytical_integral(2 * 0 + 2, 2.0 * nu_array_l0[i]))  # L=0
        for k in range(len(nu_array_l2)):
            norm_k = np.sqrt(
                analytical_integral(2 * 2 + 2, 2.0 * nu_array_l2[k])
            )  # L=2
            nu_ik = nu_array_l0[i] + nu_array_l2[k]
            # Integral for <r^0 e^-nu_i r^2 | 1/r^3 | r^2 e^-nu_k r^2>
            term_tensor = v_tensor_coeff * analytical_integral(
                2 * 0 + 2 + 2 - 3, nu_ik
            )  # Integral for r^(l+l'-1)
            c_i = c_vec_l0[i] / norm_i
            c_k = c_vec_l2[k] / norm_k
            mixing_element += c_i * c_k * term_tensor

    return mixing_element * s12_mixing


def check_virial_theorem(c_vec, nu_array, sys, l):
    """
    Calculates the Virial ratio: 2 <T> / < (4 * alpha_s / 3r) + b * r >
    If the basis is complete and well-optimized, this ratio must be identically 1.0.

    Parameters
    ----------
    c_vec : numpy.ndarray
        Eigenvector coefficients.
    nu_array : numpy.ndarray
        Gaussian basis widths.
    sys : QuarkoniumSystem
        Quarkonium system representation object.
    l : int
        Orbital angular momentum quantum number.

    Returns
    -------
    float
        The computed virial ratio.
    """
    t_exp = 0.0
    v_coulomb_exp = 0.0
    v_linear_exp = 0.0

    for i in range(len(nu_array)):
        norm_i = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array[i]))
        for k in range(len(nu_array)):
            norm_k = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array[k]))
            nu_ik = nu_array[i] + nu_array[k]

            c_i = c_vec[i] / norm_i
            c_k = c_vec[k] / norm_k

            # Kinetic Energy
            term1 = (l + 1.0) ** 2 * analytical_integral(2 * l, nu_ik)
            term2 = -2.0 * (l + 1.0) * nu_ik * analytical_integral(2 * l + 2, nu_ik)
            term3 = (
                4.0 * nu_array[i] * nu_array[k] * analytical_integral(2 * l + 4, nu_ik)
            )
            t_ij = (sys.hbar**2 / (2.0 * sys.mu)) * (term1 + term2 + term3)

            t_exp += c_i * c_k * t_ij

            v_coulomb_ij = (
                (4.0 / 3.0) * sys.alpha_s * analytical_integral(2 * l + 1, nu_ik)
            )
            v_linear_ij = sys.b * analytical_integral(2 * l + 3, nu_ik)

            v_coulomb_exp += c_i * c_k * v_coulomb_ij
            v_linear_exp += c_i * c_k * v_linear_ij

    return 2.0 * t_exp / (v_coulomb_exp + v_linear_exp)


def get_wfo_exact(c_i, nu_array, r_array):
    u_prime_0 = 0.0
    for c, nu in zip(c_i, nu_array):
        u_unnorm = r_array * np.exp(-nu * r_array**2)
        norm = np.sqrt(simpson(y=u_unnorm**2, x=r_array))
        u_prime_0 += c / norm
    return u_prime_0


def get_mass(
    bare_e,
    evecs,
    nu_array,
    sys,
    state_idx,
    spin,
    l,
    j=0,
    include_tensor_shift=True,
):
    """
    Returns the total mass. Hyperfine is handled variationally, but Spin-Orbit
    and Tensor interactions are evaluated perturbatively to avoid 1/r^3 collapse.

    Parameters
    ----------
    bare_e : numpy.ndarray
        Eigenvalues for the bare energy.
    evecs : numpy.ndarray
        Eigenvectors representing states.
    nu_array : numpy.ndarray
        Gaussian basis widths.
    sys : QuarkoniumSystem
        Quarkonium system representation object.
    state_idx : int
        Index for the targeted state.
    spin : int
        Spin quantum number.
    l : int
        Orbital angular momentum quantum number.
    j : int, optional
        Total angular momentum quantum number, by default 0.
    include_tensor_shift : bool, optional
        Flag to include tensor shift perturbations, by default True.

    Returns
    -------
    float
        Total perturbatively corrected mass.
    """
    bare_mass = sys.M_bare + bare_e[state_idx]
    c_vec = evecs[:, state_idx]
    so_shift = calc_so_shift_exact(c_vec, nu_array, sys, l, spin, j)
    tensor_shift = (
        calc_tensor_shift_exact(c_vec, nu_array, sys, l, spin, j)
        if include_tensor_shift
        else 0.0
    )
    return bare_mass + so_shift + tensor_shift


# Fundamental QFT Constants
ALPHA_EM = 1.0 / 137.036


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


def get_leptonic_width(mass_GeV, c_vec, nu_array, sys, e_q, l=0):
    """
    Calculates the leptonic decay width (V -> e+ e-) for vector mesons (Spin=1).
    """
    R_0_sq = calc_R_origin_sq_hypervirial(c_vec, nu_array, sys, l=l)

    # Use the high-energy running coupling constant evaluated at the heavy quark mass
    # scale rather than the long-range fitted Cornell parameter (sys.alpha_s).
    if sys.m_1 > 4.0:
        # Bottomonium (m_b ≈ 4.73 GeV)
        alpha_s_run = 0.20
    else:
        # Charmonium (m_c ≈ 1.5 GeV)
        alpha_s_run = 0.35

    # First-order perturbative QCD correction using the running coupling
    qcd_correction = 1.0 - (16.0 * alpha_s_run) / (3.0 * np.pi)

    width_GeV = (4.0 * ALPHA_EM**2 * e_q**2) / (mass_GeV**2) * R_0_sq * qcd_correction
    return width_GeV * 1e6  # Return in keV


def get_two_photon_width(mass_GeV, c_vec, nu_array, sys, e_q, l=0):
    """
    Calculates the two-photon decay width (P -> gamma gamma) for pseudoscalar mesons (Spin=0).
    """
    R_0_sq = calc_R_origin_sq_hypervirial(c_vec, nu_array, sys, l=l)

    # Use the high-energy running coupling constant evaluated at the heavy quark mass
    # scale rather than the long-range fitted Cornell parameter (sys.alpha_s).
    if sys.m_1 > 4.0:
        # Bottomonium (m_b ≈ 4.73 GeV)
        alpha_s_run = 0.20
    else:
        # Charmonium (m_c ≈ 1.5 GeV)
        alpha_s_run = 0.35

    # First-order perturbative QCD correction using the running coupling
    qcd_correction = 1.0 - (3.4 * alpha_s_run) / np.pi

    width_GeV = (3.0 * ALPHA_EM**2 * e_q**4) / (mass_GeV**2) * R_0_sq * qcd_correction
    return width_GeV * 1e6  # Return in keV


def calc_overlap_3p0_simplified(c_A, nu_A, c_B, nu_B, c_C, nu_C, l_A=0, l_B=0, l_C=0):
    """
    Calculates the phenomenological spatial overlap integral for a 3P0 hadronic decay A -> B + C.
    Exploits the GEM basis to evaluate the overlapping wavefunctions exactly:
    Overlap = Integral of r^(L_A+L_B+L_C+2) * exp(-(nu_A + nu_B + nu_C) * r^2) dr
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
    """
    Calculates the hadronic decay width (A -> B + C) using a simplified 3P0 pair-creation model.
    """
    if mass_A < mass_B + mass_C:
        return 0.0  # Kinematically forbidden (below threshold)

    # Two-body decay relative momentum p (Phase Space)
    p = np.sqrt(
        (mass_A**2 - (mass_B + mass_C) ** 2) * (mass_A**2 - (mass_B - mass_C) ** 2)
    ) / (2.0 * mass_A)
    overlap = calc_overlap_3p0_simplified(
        c_A, nu_A, c_B, nu_B, c_C, nu_C, l_A=l_A, l_B=0, l_C=0
    )

    # Phenomenological 3P0 width formula (Simplified Kinematics)
    width = (
        2.0
        * np.pi
        * p
        * (np.sqrt(mass_B**2 + p**2) * np.sqrt(mass_C**2 + p**2) / mass_A)
        * (gamma_3p0**2)
        * (overlap**2)
    )
    return width * 1000.0  # Return in MeV
