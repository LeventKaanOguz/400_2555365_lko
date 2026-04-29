import numpy as np
from scipy.linalg import eigh
from scipy.special import gamma


def analytical_integral(p, nu):
    r"""Helper function for analytical Gaussian integrals: \int_0^\infty r^p e^{-nu r^2} dr"""
    return 0.5 * gamma((p + 1.0) / 2.0) / (nu ** ((p + 1.0) / 2.0))


class QuarkoniumSystem:
    def __init__(self, m_1, m_2, alpha_s, b, c, sigma_smear=None, hbar=1.0):
        self.m_1 = m_1
        self.m_2 = m_2
        self.mu = (m_1 * m_2) / (m_1 + m_2)
        self.M_bare = m_1 + m_2
        self.alpha_s = alpha_s
        self.b = b
        self.c = c
        if sigma_smear is None:
            # Tie sigma_smear to the reduced mass for better physical consistency
            self.sigma_smear = 1.0 * np.sqrt(self.mu)
        else:
            self.sigma_smear = sigma_smear
        self.hbar = hbar


def solve_gem(sys: QuarkoniumSystem, r, l, spin=0, n_basis=25, r_min=0.05, r_max=15.0):
    nu_array = np.zeros(n_basis)
    ratio = (r_max / r_min) ** (2.0 / (n_basis - 1)) if n_basis > 1 else 1.0
    for i in range(n_basis):
        nu_array[i] = 1.0 / (r_min**2 * ratio**i)

    # --- Spin-Dependent Variational Constants ---
    spin_dot = -0.75 if spin == 0 else 0.25
    hf_coeff = (
        (32.0 * np.pi * sys.alpha_s)
        / (9.0 * sys.m_1 * sys.m_2)
        * (sys.sigma_smear / np.sqrt(np.pi)) ** 3
    )

    # 1. Calculate Exact Analytical Norms for the unnormalized basis
    norms = np.sqrt(analytical_integral(2 * l + 2, 2.0 * nu_array))

    # 2. Build Analytical Matrices using Gamma Functions
    nu_ij = nu_array[:, np.newaxis] + nu_array[np.newaxis, :]

    # Core analytical integrals (vectorized)
    I_2L = analytical_integral(2 * l, nu_ij)
    I_2L1 = analytical_integral(2 * l + 1, nu_ij)
    I_2L2 = analytical_integral(2 * l + 2, nu_ij)
    I_2L3 = analytical_integral(2 * l + 3, nu_ij)
    I_2L4 = analytical_integral(2 * l + 4, nu_ij)

    # Overlap Matrix
    S_ij = I_2L2

    # Kinetic Energy (T) Matrix via exact first derivative expansion
    term1 = (l + 1.0) ** 2 * I_2L
    term2 = -2.0 * (l + 1.0) * nu_ij * I_2L2
    term3 = 4.0 * nu_array[:, np.newaxis] * nu_array[np.newaxis, :] * I_2L4
    T_ij = (sys.hbar**2 / (2.0 * sys.mu)) * (term1 + term2 + term3)

    # Potential Energy (V) Matrix
    V_coulomb = -(4.0 / 3.0) * sys.alpha_s * I_2L1
    V_linear = sys.b * I_2L3
    V_const = sys.c * S_ij

    # Centrifugal barrier matrix for L > 0
    V_cent = 0.0
    if l > 0:
        V_cent = (l * (l + 1.0) * sys.hbar**2 / (2.0 * sys.mu)) * I_2L

    # --- Variational Spin-Dependent Interactions ---
    V_hf = 0.0
    if l == 0:  # Hyperfine contact interaction primarily affects S-waves
        V_hf = (
            hf_coeff
            * spin_dot
            * analytical_integral(2 * l + 2, nu_ij + sys.sigma_smear**2)
        )

    # Total Hamiltonian element
    H_ij = T_ij + V_coulomb + V_linear + V_const + V_cent + V_hf

    # Instantly normalize the matrices for perfect numerical conditioning
    norm_matrix = norms[:, np.newaxis] * norms[np.newaxis, :]
    H = H_ij / norm_matrix
    S = S_ij / norm_matrix

    # 3. Solve the well-conditioned analytical eigenvalue problem
    try:
        evals, evecs = eigh(H, S)
    except np.linalg.LinAlgError:
        S += np.eye(n_basis) * 1e-12
        evals, evecs = eigh(H, S)

    # 4. Reconstruct the physical spatial wavefunctions (strictly for observables/plotting)
    u_gem = np.zeros((len(r), len(evals)))
    for k in range(len(evals)):
        c_k = evecs[:, k]
        wavefunc = np.zeros_like(r)
        for i in range(n_basis):
            u_i = r ** (l + 1) * np.exp(-nu_array[i] * r**2)
            # The evecs of a normalized S matrix guarantee the sum is analytically normalized
            wavefunc += c_k[i] * (u_i / norms[i])

        u_gem[:, k] = wavefunc

    return evals, u_gem, evecs, nu_array
