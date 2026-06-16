"""Print the derived relativistic theory systematic per sector / level.

A diagnostic wrapper around observables.calc_relativistic_shift: the theory
uncertainty on each level is the magnitude of the leading omitted relativistic
correction Delta H = -p^4/(8c^2)(1/m1^3 + 1/m2^3), the next term in the v^2/c^2
expansion. Also reports the NRQCD velocity <v^2> (the expansion parameter). This
is the model-viability bar: the per-state sigma the *fit weight* uses as a
well-posed regularizer. It is NOT the chi-square / pull denominator -- that uses
the computational sigma (experimental error in quadrature with the propagated
parameter covariance). Reads the fitted Cornell parameters from
results/<sector>/params.csv.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quarkonia.gem_solver import QuarkoniumSystem, solve_gem
from quarkonia.observables import calc_relativistic_shift
from quarkonia import paths
import pandas as pd

# Quark masses (GeV), mirroring run_spectrum.py.
SECTORS = {
    "bb": (4.730, 4.730),
    "cc": (1.500, 1.500),
    "bc": (4.730, 1.500),
}
# (label, l, spin, n_idx)
LEVELS = [("1S", 0, 1, 0), ("2S", 0, 1, 1), ("1P", 1, 1, 0)]


def main():
    r = np.linspace(1e-4, 20, 4000)
    print(f"{'sector':6} {'level':5} {'<v^2>':>8} {'sqrt<p^2>(GeV)':>15} "
          f"{'sigma_theory (MeV)':>18}")
    print("-" * 56)
    for sec, (m1, m2) in SECTORS.items():
        p = paths.params_csv(sec)
        if not os.path.exists(p):
            print(f"{sec}: no params.csv -- run run_spectrum.py first")
            continue
        row = pd.read_csv(p).iloc[0]
        s = QuarkoniumSystem(m1, m2, float(row["alpha_s"]), float(row["b"]),
                             float(row["c"]), sigma_smear=float(row["sigma"]))
        for label, l, spin, n in LEVELS:
            _, _, evec, nu = solve_gem(s, r, l=l, spin=spin)
            if n >= evec.shape[1]:
                continue
            dE, v2 = calc_relativistic_shift(evec[:, n], nu, s, l)
            p2 = v2 * min(m1, m2) ** 2
            print(f"{sec:6} {label:5} {v2:8.3f} {np.sqrt(p2):15.3f} "
                  f"{dE * 1000:18.1f}")


if __name__ == "__main__":
    main()
