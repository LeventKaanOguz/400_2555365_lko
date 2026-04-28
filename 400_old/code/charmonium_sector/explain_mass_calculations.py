#!/usr/bin/env python3

import os
import sys
import io
import contextlib


def print_header(title):
    print("\n" + "=" * 85)
    print(f" {title.center(83)} ")
    print("=" * 85)


def print_sub_header(title):
    print("\n" + "-" * 85)
    print(f" {title} ")
    print("-" * 85)


def explain_theory():
    print_header("THEORETICAL FRAMEWORK: PHENOMENOLOGICAL MASS CALCULATION")
    print("""
The total physical mass (M) of a meson state in our model is calculated using 
the constituent quark mass and three distinct energy contributions:

    M = 2 * m_q  +  E_bare(N, L)  +  ΔE_HF(L, S)  +  ΔE_SO(L, S, J)

--- 1. Constituent Mass (2 * m_q) ---
The sum of the rest masses of the charm quark (m_c) and anti-charm quark (m_c-bar).

--- 2. Bare Energy: E_bare(N, L) ---
The energy eigenvalue from solving the radial Schrödinger equation. This is done
via the Gaussian Expansion Method (GEM), which solves the generalized eigenvalue
problem Hc = ESc. The Hamiltonian and Overlap matrices (H, S) are built from a
non-orthogonal basis of Gaussian functions: u_i(r) = r^(L+1) * exp(-ν_i * r²).

The potential used to find the bare energy includes the central Cornell potential 
and the centrifugal barrier term for states with L > 0:
    V_eff(r) = -(4/3) * α_s / r  +  b * r  +  c  +  L(L+1)ℏ² / (2μr²)

--- 3. Hyperfine Splitting: ΔE_HF(L, S) ---
A spin-spin contact interaction that separates Singlet (S=0) and Triplet (S=1) states.
In our model, the Dirac delta function δ(r) is "smeared" with a Gaussian to 
avoid numerical instability, making it non-zero over a small region. This term
primarily affects S-waves (L=0) where the wavefunction is non-zero at the origin.

The energy shift is the expectation value of the hyperfine potential:
    ΔE_HF = < V_HF > = ∫ u(r)² * V_HF(r) dr

Where the potential is:
    V_HF(r) = [ (32πα_s)/(9m_q²) * (S₁·S₂) ] * [ (σ/√π)³ * exp(-σ²r²) ]

    * If S = 0 (Singlet): S_1·S_2 = -3/4  (Energy goes down)
    * If S = 1 (Triplet): S_1·S_2 = +1/4  (Energy goes up)
    * If L > 0          : ΔE_HF = 0       (Wavefunction is 0 at the origin)

--- 4. Spin-Orbit Splitting: ΔE_SO(L, S, J) ---
Fine structure splitting caused by the interaction between the orbital angular 
momentum (L) and total spin (S). It only affects states where both L > 0 and S > 0.

The energy shift is the expectation value of the spin-orbit potential:
    ΔE_SO = < V_SO > = <L·S> * ∫ u(r)² * V_LS(r) dr

Where the operators and potential are:
    L·S     = 1/2 * [ J(J+1) - L(L+1) - S(S+1) ]
    V_LS(r) = 1/(2m_q²) * ( (4α_s)/r³ - b/r )
""")


def explain_states(results=None):
    print_header("STATE-BY-STATE BREAKDOWN")

    mass_dict = {}
    if results and "comparison_table_data" in results:
        mass_dict = {row[0]: row[1] for row in results["comparison_table_data"]}

    states = [
        {"name": "(1^1S) η_c", "n": 1, "L": 0, "S": 0, "J": 0, "bare": "evals_gem[0]"},
        {"name": "(1^3S) J/ψ", "n": 1, "L": 0, "S": 1, "J": 1, "bare": "evals_gem[0]"},
        {
            "name": "(1^1P) h_c",
            "n": 1,
            "L": 1,
            "S": 0,
            "J": 1,
            "bare": "evals_gem_1[0]",
        },
        {
            "name": "(1^3P_0) χ_c0",
            "n": 1,
            "L": 1,
            "S": 1,
            "J": 0,
            "bare": "evals_gem_1[0]",
        },
        {
            "name": "(1^3P_1) χ_c1",
            "n": 1,
            "L": 1,
            "S": 1,
            "J": 1,
            "bare": "evals_gem_1[0]",
        },
        {
            "name": "(1^3P_2) χ_c2",
            "n": 1,
            "L": 1,
            "S": 1,
            "J": 2,
            "bare": "evals_gem_1[0]",
        },
        {
            "name": "(1^1D) η_c2",
            "n": 1,
            "L": 2,
            "S": 0,
            "J": 2,
            "bare": "evals_gem_2[0]",
        },
        {"name": "(1^3D) ψ", "n": 1, "L": 2, "S": 1, "J": 1, "bare": "evals_gem_2[0]"},
        {"name": "(2^1S) η_c", "n": 2, "L": 0, "S": 0, "J": 0, "bare": "evals_gem[1]"},
        {
            "name": "(2^3S) ψ(2S)",
            "n": 2,
            "L": 0,
            "S": 1,
            "J": 1,
            "bare": "evals_gem[1]",
        },
        {
            "name": "(2^1P) h_c",
            "n": 2,
            "L": 1,
            "S": 0,
            "J": 1,
            "bare": "evals_gem_1[1]",
        },
        {
            "name": "(2^3P_0) χ_c0",
            "n": 2,
            "L": 1,
            "S": 1,
            "J": 0,
            "bare": "evals_gem_1[1]",
        },
        {
            "name": "(2^3P_1) χ_c1",
            "n": 2,
            "L": 1,
            "S": 1,
            "J": 1,
            "bare": "evals_gem_1[1]",
        },
        {
            "name": "(2^3P_2) χ_c2",
            "n": 2,
            "L": 1,
            "S": 1,
            "J": 2,
            "bare": "evals_gem_1[1]",
        },
        {
            "name": "(2^1D) η_c2",
            "n": 2,
            "L": 2,
            "S": 0,
            "J": 2,
            "bare": "evals_gem_2[1]",
        },
        {"name": "(2^3D) ψ", "n": 2, "L": 2, "S": 1, "J": 1, "bare": "evals_gem_2[1]"},
    ]

    for state in states:
        name = state["name"]
        N = state["n"]
        L = state["L"]
        S = state["S"]
        J = state["J"]
        bare = state["bare"]

        print_sub_header(f"State: {name} [N={N}, L={L}, S={S}, J={J}]")

        # 1. Bare Energy
        wave_name = "S" if L == 0 else "P" if L == 1 else "D"
        print(f"[*] Bare Energy      : Read from {bare} (The {N}{wave_name} state)")

        # 2. Hyperfine Shift
        if L == 0:
            if S == 0:
                print(
                    f"[*] Hyperfine (HF)   : CALCULATED. L=0 allows contact. S=0 means S_1·S_2 = -3/4. Shifts Mass DOWN."
                )
            else:
                print(
                    f"[*] Hyperfine (HF)   : CALCULATED. L=0 allows contact. S=1 means S_1·S_2 = +1/4. Shifts Mass UP."
                )
        else:
            print(
                f"[*] Hyperfine (HF)   : ZERO. Wavefunction for L={L} vanishes at origin (r=0). Contact interaction is 0."
            )

        # 3. Spin-Orbit Shift
        if L == 0:
            print(
                f"[*] Spin-Orbit (SO)  : ZERO. Orbital angular momentum L=0, so L·S = 0."
            )
        elif S == 0:
            print(
                f"[*] Spin-Orbit (SO)  : ZERO. Spin angular momentum S=0, so L·S = 0."
            )
        else:
            ls_dot = 0.5 * (J * (J + 1) - L * (L + 1) - S * (S + 1))
            print(
                f"[*] Spin-Orbit (SO)  : CALCULATED. L={L}, S={S}, J={J}. L·S factor = {ls_dot}."
            )

        # 4. Final Formula
        components = ["2*m_c", bare]
        if L == 0:
            components.append("ΔE_HF")
        if L > 0 and S > 0:
            components.append("ΔE_SO")

        print(f"[*] Total Mass Eq    : M = " + " + ".join(components))
        if name in mass_dict:
            print(f"[*] Final Calc Mass  : {mass_dict[name]:.4f} GeV")


def explain_findings(results):
    if not results or "comparison_table_data" not in results:
        return

    print_header("FINAL NUMERICAL FINDINGS (CHARMONIUM SECTOR)")
    print(
        "The following table summarizes the calculated masses against experimental data"
    )
    print("as outputted by the charmonium sector pipeline:\n")

    print(
        f"{'State':<15} | {'Calculated [GeV]':<18} | {'Experimental [GeV]':<18} | {'Error [MeV]'}"
    )
    print("-" * 75)
    for row in results["comparison_table_data"]:
        state = row[0]
        calc_mass = row[1]
        exp_str = row[4]
        if exp_str != "-":
            exp_mass = float(exp_str.split("±")[0])
            err_mev = (calc_mass - exp_mass) * 1000.0
            print(
                f"{state:<15} | {calc_mass:<18.4f} | {exp_mass:<18.4f} | {err_mev:+.1f}"
            )
        else:
            print(f"{state:<15} | {calc_mass:<18.4f} | {'-':<18} | -")


def explain_code_implementation():
    print_header("HOW THIS IS IMPLEMENTED IN THE PIPELINE (get_mass)")
    print("""
In `charmonium_sector.py`, this entire logic is compactly handled by the `get_mass` function:

    def get_mass(evals, u_arr, state_idx, spin, l, j=None):
        bare = evals[state_idx]
        
        # Hyperfine splitting is zero for L > 0 since wavefunctions vanish at the origin
        hf_shift = calc_hf_shift(u_arr[:, state_idx], spin=spin) if l == 0 else 0.0

        so_shift = 0.0
        # Spin orbit requires both orbital angular momentum and spin
        if l > 0 and spin > 0 and j is not None:
            so_shift = calc_so_shift(u_arr[:, state_idx], l=l, s=spin, j=j)

        return 2 * M_Q + bare + hf_shift + so_shift

    * The `state_idx` tells it whether to take the 1st (index 0) or 2nd (index 1) eigenvalue from the GEM array.
    * The `calc_hf_shift` integrates the wavefunction over the smeared Gaussian delta distribution.
    * The `calc_so_shift` integrates the wavefunction over the typical V_LS(r) potential.
""")


def main():
    try:
        print("Gathering live data from charmonium_sector.py... (Please wait)\n")
        sys.path.append(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "charmonium_sector")
            )
        )
        try:
            from charmonium_sector import run_comparisons

            with contextlib.redirect_stdout(io.StringIO()):
                results = run_comparisons()
        except ImportError:
            results = None
            print("Note: Could not import charmonium_sector.py to fetch live results.")

        explain_theory()
        explain_states(results)
        explain_findings(results)
        explain_code_implementation()
        print("\n" + "=" * 85)
        print(" Explanation generation complete. ".center(85))
        print("=" * 85 + "\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
