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


def check_virial_theorem(c_vec, nu_array, sys, l, spin=None):
    """
    Calculates the quantum Virial ratio  2<T> / <r dV/dr>.

    For a bound eigenstate the virial theorem  2<T> = <r dV/dr>  holds exactly,
    so a converged variational basis must return 1.0. The full Cornell-plus-
    hyperfine potential is used:

        V(r)      = -(4/3) alpha_s / r + b r + c + V_hf(r)
        r dV/dr   =  (4/3) alpha_s / r + b r + r dV_hf/dr

    with the Gaussian-smeared contact hyperfine term
    V_hf(r) = K exp(-sigma^2 r^2),  r dV_hf/dr = -2 sigma^2 r^2 K exp(-sigma^2 r^2).

    The constant ``c`` does not contribute (dc/dr = 0). Earlier versions omitted
    the hyperfine piece, which made the ratio spuriously deviate from 1 (e.g.
    ~1.2 for the strongly-smeared singlet S-wave) even when the basis was fully
    converged. Passing ``spin`` restores the correct, physically complete check.

    Parameters
    ----------
    c_vec : numpy.ndarray
        Eigenvector coefficients (un-normalized GEM coefficients).
    nu_array : numpy.ndarray
        Gaussian basis widths.
    sys : QuarkoniumSystem
        Quarkonium system representation object.
    l : int
        Orbital angular momentum quantum number.
    spin : int or None, optional
        Total quark spin (0 or 1). Required to include the hyperfine term for
        S-waves; if None the hyperfine contribution is omitted (Coulomb+linear
        only), matching the legacy behaviour.

    Returns
    -------
    float
        The computed virial ratio (1.0 for a fully converged eigenstate).
    """
    norms = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array))
    c_norm = c_vec / norms
    nu_ij = nu_array[:, np.newaxis] + nu_array[np.newaxis, :]

    # Kinetic energy  <T>  via the exact first-derivative expansion
    term1 = (l + 1.0) ** 2 * analytical_integral(2 * l, nu_ij)
    term2 = -2.0 * (l + 1.0) * nu_ij * analytical_integral(2 * l + 2, nu_ij)
    term3 = (
        4.0
        * nu_array[:, np.newaxis]
        * nu_array[np.newaxis, :]
        * analytical_integral(2 * l + 4, nu_ij)
    )
    T = (sys.hbar**2 / (2.0 * sys.mu)) * (term1 + term2 + term3)
    t_exp = c_norm @ T @ c_norm

    # <r dV/dr> for the Coulomb + linear Cornell potential
    rdv_coulomb = (4.0 / 3.0) * sys.alpha_s * analytical_integral(2 * l + 1, nu_ij)
    rdv_linear = sys.b * analytical_integral(2 * l + 3, nu_ij)
    rdv = c_norm @ (rdv_coulomb + rdv_linear) @ c_norm

    # Hyperfine contribution to <r dV/dr> (S-waves only, requires spin)
    if spin is not None and l == 0:
        spin_dot = -0.75 if spin == 0 else 0.25
        hf_coeff = (
            (32.0 * np.pi * sys.alpha_s)
            / (9.0 * sys.m_1 * sys.m_2)
            * (sys.sigma_smear / np.sqrt(np.pi)) ** 3
        )
        k_hf = hf_coeff * spin_dot
        s2 = sys.sigma_smear**2
        rdv_hf = -2.0 * s2 * k_hf * analytical_integral(2 * l + 4, nu_ij + s2)
        rdv += c_norm @ rdv_hf @ c_norm

    return 2.0 * t_exp / rdv


def calc_relativistic_shift(c_vec, nu_array, sys, l):
    r"""Leading relativistic correction to a level, and the NRQCD velocity.

    A non-relativistic potential model omits the next term in expanding
    :math:`\sqrt{p^2c^2+m^2c^4}`, namely

    .. math:: \Delta H = -\frac{p^4}{8c^2}\left(\frac1{m_1^3}+\frac1{m_2^3}\right).

    Its expectation value on the eigenstate is a *computed* (not assumed) estimate
    of how far that level can be expected to sit from experiment -- it is the
    "size of the first omitted term", the standard effective-theory truncation
    error. We use ``abs(dE_rel)`` as the per-state theory uncertainty on the mass.

    ``<p^4>`` is evaluated exactly in the non-orthogonal Gaussian basis via
    ``<p^4> = c^T P2 S^{-1} P2 c`` with ``P2 = 2 mu T`` (the model's own kinetic
    operator), so the spread of ``p^2`` is kept rather than approximated by
    ``<p^2>^2``. The convention matches :func:`check_virial_theorem`: ``c_vec`` are
    the GEM coefficients on the normalized basis and ``c_norm = c_vec / norms``.

    Parameters
    ----------
    c_vec : numpy.ndarray
        Eigenvector coefficients (un-normalized GEM coefficients).
    nu_array : numpy.ndarray
        Gaussian basis widths.
    sys : QuarkoniumSystem
        Quarkonium system (provides ``mu``, ``m_1``, ``m_2``, ``hbar``).
    l : int
        Orbital angular momentum quantum number.

    Returns
    -------
    (float, float)
        ``(abs(dE_rel) in GeV, <v^2>)`` where ``<v^2> = <p^2> / m_light^2`` is the
        velocity-squared of the lighter quark (the expansion parameter).
    """
    norms = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array))
    c_norm = c_vec / norms
    nu_ij = nu_array[:, np.newaxis] + nu_array[np.newaxis, :]

    # Kinetic operator T (same exact first-derivative expansion as solve_gem /
    # check_virial_theorem); p^2 = 2 mu T.
    term1 = (l + 1.0) ** 2 * analytical_integral(2 * l, nu_ij)
    term2 = -2.0 * (l + 1.0) * nu_ij * analytical_integral(2 * l + 2, nu_ij)
    term3 = (
        4.0
        * nu_array[:, np.newaxis]
        * nu_array[np.newaxis, :]
        * analytical_integral(2 * l + 4, nu_ij)
    )
    T = (sys.hbar**2 / (2.0 * sys.mu)) * (term1 + term2 + term3)
    P2 = 2.0 * sys.mu * T
    S = analytical_integral(2 * l + 2, nu_ij)  # raw overlap

    p2 = float(c_norm @ P2 @ c_norm)
    # exact <p^4> via the resolution of identity in the non-orthogonal basis
    p4 = float(c_norm @ P2 @ np.linalg.solve(S, P2 @ c_norm))

    dE_rel = -p4 / 8.0 * (1.0 / sys.m_1**3 + 1.0 / sys.m_2**3)  # c = 1 (GeV units)
    m_light = min(sys.m_1, sys.m_2)
    v2 = p2 / m_light**2
    return abs(dE_rel), v2


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
