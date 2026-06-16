#!/usr/bin/env python3
"""
============================== DEPRECATED ==============================
This leave-one-out (LOO) cross-validation is **deprecated** and no longer
part of the pipeline. It is kept runnable for reference only -- it is not
called by run_spectrum.py, is omitted from REPORT.md and the figures, and
its output (summary/cross_validation.csv) is no longer consumed.

Rationale: the goodness-of-fit chi-square now uses the *computational*
sigma (experimental error in quadrature with the propagated parameter
covariance), which is itself a direct out-of-sample-style validation of
whether the implementation reproduces experiment within its computed
precision. The separate LOO refit is therefore redundant.
=======================================================================

Cross-validation overfitting check for the Cornell-potential mass fit.

A 4-parameter fit to ~12 masses *could* be memorising the training levels instead
of learning the dynamics. The clean way to find out is the train/test idea taken to
its rigorous, small-sample limit: **leave-one-out (LOO)**. For each measured level we
refit the potential on every *other* level and then predict the held-out one. The
RMS of those held-out predictions (``loo_rms``) is the model's honest out-of-sample
error. Compared with the in-sample full-fit RMS (``full_rms``):

* ``loo_rms`` ≈ ``full_rms``  -> the rigid, physically-constrained Cornell form
  generalises; it is *not* overfitting.
* ``loo_rms`` >> ``full_rms``  -> removing a single level swings the prediction a
  lot, i.e. the parameters are memorising rather than learning.

LOO is preferred here over a 2-fold split because each fit keeps N-1 well-anchored
levels (a half-split can accidentally drop the ground-state anchors and explode for
reasons that have nothing to do with overfitting).

The GEM eigenvalues are built from analytical integrals over the Gaussian widths and
are independent of the radial grid (which only shapes the output wavefunctions), so a
small grid is used here for speed without changing the fit.

Usage
-----
    conda run -n 400 python scripts/cross_validate.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from quarkonia import paths
from quarkonia.fitter import residuals, compute_spectrum_masses

# Sector fit configuration (mirrors run_spectrum, masses only).
SECTORS = [
    {
        "id": "bb",
        "name": "Bottomonium (b_bbar)",
        "m_1": 4.730,
        "m_2": 4.730,
        "x0": [0.350, 0.193, 0.030, 1.34],
        "bounds": ([0.1, 0.1, -1.0, 0.3], [0.8, 0.35, 1.0, 5.0]),
    },
    {
        "id": "cc",
        "name": "Charmonium (c_cbar)",
        "m_1": 1.500,
        "m_2": 1.500,
        "x0": [0.400, 0.183, -0.250, 1.09],
        "bounds": ([0.1, 0.1, -1.0, 0.3], [0.8, 0.35, 1.0, 5.0]),
    },
]


def _fit(cfg, masses, mass_err, r):
    """Least-squares fit on the given (subset of) masses; return parameters."""
    result = least_squares(
        residuals,
        cfg["x0"],
        args=(cfg["m_1"], cfg["m_2"], masses, mass_err, r, None, 0.0),
        bounds=cfg["bounds"],
        method="trf",
    )
    return result.x


def _rms_mev(params, cfg, masses, r):
    """RMS mass deviation (MeV) of the model on the given labels."""
    pred = compute_spectrum_masses(params, cfg["m_1"], cfg["m_2"], r)
    diffs = [
        (pred[label] - exp) * 1000.0
        for label, exp in masses.items()
        if exp is not None and label in pred
    ]
    return float(np.sqrt(np.mean(np.square(diffs)))) if diffs else float("nan")


def run():
    from quarkonia.pdg_loader import load_pdg_data

    all_pdg = load_pdg_data()
    # Eigenvalues are grid-independent; a small grid keeps the many refits fast.
    r = np.linspace(0.02, 15.0, 200)

    rows = []
    for cfg in SECTORS:
        sid = cfg["id"]
        full_masses = {k: v for k, v in all_pdg[sid].items() if v is not None}
        mass_err = all_pdg.get(f"{sid}_mass_err_GeV", {})
        # Only keep levels the model actually predicts.
        probe = compute_spectrum_masses(cfg["x0"], cfg["m_1"], cfg["m_2"], r)
        labels = [k for k in full_masses if k in probe]
        labels.sort()

        # In-sample reference: fit on every level.
        p_full = _fit(cfg, full_masses, mass_err, r)
        full_rms = _rms_mev(p_full, cfg, full_masses, r)

        # Leave-one-out: predict each level from a fit on all the others.
        held_out_sq = []
        for held in labels:
            train = {k: v for k, v in full_masses.items() if k != held}
            p = _fit(cfg, train, mass_err, r)
            pred = compute_spectrum_masses(p, cfg["m_1"], cfg["m_2"], r)
            held_out_sq.append(((pred[held] - full_masses[held]) * 1000.0) ** 2)
        loo_rms = float(np.sqrt(np.mean(held_out_sq)))

        rows.append(
            {
                "sector": cfg["name"],
                "n_levels": len(labels),
                "full_rms_mev": full_rms,
                "loo_rms_mev": loo_rms,
                "loo_over_full": loo_rms / full_rms if full_rms > 0 else float("nan"),
            }
        )
        print(
            f"{cfg['name']:<22} | levels={len(labels):2d} | "
            f"full-fit RMS={full_rms:6.2f} | LOO RMS={loo_rms:6.2f} | "
            f"LOO/full={loo_rms / full_rms:4.2f}"
        )

    df = pd.DataFrame(rows)
    out = paths.summary_csv("cross_validation.csv")
    df.to_csv(out, index=False)
    print(f"\nCross-validation summary saved to {out}")


if __name__ == "__main__":
    print("--- Cross-Validation (train/test split on mass levels) ---")
    run()
