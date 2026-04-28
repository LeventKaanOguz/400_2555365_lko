import numpy as np
from scipy.integrate import simpson


def calc_hf_shift(u_array, r, sys, spin=0):
    spin_dot = -0.75 if spin == 0 else 0.25
    hf_coeff = (32.0 * np.pi * sys.alpha_s) / (9.0 * sys.m_1 * sys.m_2)
    smeared_delta = (sys.sigma_smear / np.sqrt(np.pi)) ** 3 * np.exp(
        -(sys.sigma_smear**2) * r**2
    )
    return simpson(y=(u_array**2) * hf_coeff * spin_dot * smeared_delta, x=r)


def calc_so_shift(u_array, r, sys, l, s, j):
    if l == 0 or s == 0:
        return 0.0
    ls_dot = 0.5 * (j * (j + 1) - l * (l + 1) - s * (s + 1))

    term1 = (
        0.25
        * (1.0 / sys.m_1**2 + 1.0 / sys.m_2**2)
        * ((4.0 * sys.alpha_s) / (3.0 * r**3) - sys.b / r)
    )
    term2 = (1.0 / (sys.m_1 * sys.m_2)) * ((4.0 * sys.alpha_s) / (3.0 * r**3))
    v_ls = term1 + term2
    return simpson(y=(u_array**2) * v_ls, x=r) * ls_dot


def calc_tensor_shift(u_array, r, sys, l, s, j):
    if l == 0 or s == 0:
        return 0.0
    if s == 1:
        if j == l - 1:
            s12 = -(l + 1.0) / (2.0 * l - 1.0)
        elif j == l:
            s12 = 1.0
        elif j == l + 1:
            s12 = -l / (2.0 * l + 3.0)
        else:
            s12 = 0.0
    else:
        s12 = 0.0

    v_tensor = (4.0 * sys.alpha_s) / (3.0 * sys.m_1 * sys.m_2 * r**3)
    return simpson(y=(u_array**2) * v_tensor, x=r) * s12


def get_wfo_exact(c_i, nu_array, r_array):
    u_prime_0 = 0.0
    for c, nu in zip(c_i, nu_array):
        u_unnorm = r_array * np.exp(-nu * r_array**2)
        norm = np.sqrt(simpson(y=u_unnorm**2, x=r_array))
        u_prime_0 += c / norm
    return u_prime_0


def get_mass(bare_e, u_arr, r, sys, state_idx, spin, l, j=None):
    bare_mass = sys.M_bare + bare_e[state_idx]
    hf_shift = calc_hf_shift(u_arr[:, state_idx], r, sys, spin=spin) if l == 0 else 0.0
    so_shift = 0.0
    tensor_shift = 0.0

    if l > 0 and spin > 0 and j is not None:
        so_shift = calc_so_shift(u_arr[:, state_idx], r, sys, l=l, s=spin, j=j)
        tensor_shift = calc_tensor_shift(u_arr[:, state_idx], r, sys, l=l, s=spin, j=j)

    return bare_mass + hf_shift + so_shift + tensor_shift
