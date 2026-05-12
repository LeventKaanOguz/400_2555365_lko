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
    # For L=0, L'=2, the exact S12 matrix element is sqrt(8).
    s12_mixing = np.sqrt(8.0)

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


def calc_rms_radius(c_vec, nu_array, l=0):
    """
    Calculates the root-mean-square (RMS) radius sqrt(<r^2>) of a state.
    """
    r2_exp = 0.0
    for i in range(len(nu_array)):
        norm_i = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array[i]))
        for k in range(len(nu_array)):
            norm_k = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array[k]))
            nu_ik = nu_array[i] + nu_array[k]

            term_r2 = analytical_integral(2 * l + 4, nu_ik)

            c_i = c_vec[i] / norm_i
            c_k = c_vec[k] / norm_k
            r2_exp += c_i * c_k * term_r2

    return np.sqrt(r2_exp)
