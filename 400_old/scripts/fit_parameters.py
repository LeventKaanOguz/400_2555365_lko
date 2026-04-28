#!/usr/bin/env python3

import os
import sys
import numpy as np
from scipy.optimize import least_squares
import contextlib
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.pipeline import run_cornell_pipeline

# Experimental PDG masses for Bottomonium
EXP_MASSES = {
    "(1^1S) η_b": 9.3987,
    "(1^3S) Υ_b": 9.4603,
    "(1^1P) h_b": 9.8993,
    "(1^3P_0) χ_b0": 9.8594,
    "(1^3P_1) χ_b1": 9.8928,
    "(1^3P_2) χ_b2": 9.9122,
    "(2^3S) Υ": 10.0233,
    "(2^1P) h_b": 10.2598,
    "(2^3P_0) χ_b0": 10.2325,
    "(2^3P_1) χ_b1": 10.2555,
    "(2^3P_2) χ_b2": 10.2687,
}


def residuals(params):
    alpha_s, b, c = params
    m_q = 4.730  # Locked constituent quark mass for Bottomonium

    with contextlib.redirect_stdout(io.StringIO()):
        results = run_cornell_pipeline(
            m_q=m_q,
            alpha_s=alpha_s,
            b=b,
            c=c,
            sigma_smear=1.5,
            exp_mass_1s=EXP_MASSES["(1^1S) η_b"],
            exp_mass_1s_triplet=EXP_MASSES["(1^3S) Υ_b"],
            exp_mass_2s=EXP_MASSES["(2^3S) Υ"],
            hbar=1.0,
            system_name="Fit",
        )

    evals_gem = results["evals_gem"]
    u_gem = results["u_gem"]
    evals_gem_1 = results["evals_gem_1"]
    u_gem_1 = results["u_gem_1"]
    calc_hf_shift = results["calc_hf_shift"]
    calc_so_shift = results["calc_so_shift"]
    calc_tensor_shift = results.get("calc_tensor_shift", lambda u, l, s, j: 0.0)

    def get_mass(evals, u_arr, state_idx, spin, l, j=None):
        bare = evals[state_idx]
        hf_shift = calc_hf_shift(u_arr[:, state_idx], spin=spin) if l == 0 else 0.0
        so_shift = 0.0
        tensor_shift = 0.0
        if l > 0 and spin > 0 and j is not None:
            so_shift = calc_so_shift(u_arr[:, state_idx], l=l, s=spin, j=j)
            tensor_shift = calc_tensor_shift(u_arr[:, state_idx], l=l, s=spin, j=j)
        return 2 * m_q + bare + hf_shift + so_shift + tensor_shift

    calculated = {
        "(1^1S) η_b": get_mass(evals_gem, u_gem, 0, 0, 0),
        "(1^3S) Υ_b": get_mass(evals_gem, u_gem, 0, 1, 0),
        "(1^1P) h_b": get_mass(evals_gem_1, u_gem_1, 0, 0, 1),
        "(1^3P_0) χ_b0": get_mass(evals_gem_1, u_gem_1, 0, 1, 1, j=0),
        "(1^3P_1) χ_b1": get_mass(evals_gem_1, u_gem_1, 0, 1, 1, j=1),
        "(1^3P_2) χ_b2": get_mass(evals_gem_1, u_gem_1, 0, 1, 1, j=2),
        "(2^3S) Υ": get_mass(evals_gem, u_gem, 1, 1, 0),
        "(2^1P) h_b": get_mass(evals_gem_1, u_gem_1, 1, 0, 1),
        "(2^3P_0) χ_b0": get_mass(evals_gem_1, u_gem_1, 1, 1, 1, j=0),
        "(2^3P_1) χ_b1": get_mass(evals_gem_1, u_gem_1, 1, 1, 1, j=1),
        "(2^3P_2) χ_b2": get_mass(evals_gem_1, u_gem_1, 1, 1, 1, j=2),
    }

    res = []
    for state, exp_m in EXP_MASSES.items():
        res.append((calculated[state] - exp_m) * 1000.0)

    return res


if __name__ == "__main__":
    print("Starting Global chi^2 Parameter Fitting for Bottomonium...")
    # Initial guess: alpha_s, b, c
    initial_guess = [0.381, 0.183, 0.070]
    print(
        f"Initial Guess: alpha_s={initial_guess[0]}, b={initial_guess[1]}, c={initial_guess[2]}, m_q=4.730 (locked)"
    )

    res = least_squares(
        residuals,
        x0=initial_guess,
        bounds=([0.2, 0.1, -0.5], [0.6, 0.3, 0.5]),
        verbose=2,
    )

    print("\n--- Optimized Parameters ---")
    print(f"alpha_s = {res.x[0]:.4f}")
    print(f"b       = {res.x[1]:.4f} GeV^2")
    print(f"c       = {res.x[2]:.4f} GeV")
    print(f"m_q     = 4.7300 GeV (locked)")
    print(f"Cost (chi^2): {res.cost:.4f}")
