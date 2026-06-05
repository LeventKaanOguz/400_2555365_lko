"""Canonical locations for everything written under ``results/``.

One place decides the layout so the writers (``run_spectrum``, ``metrics``) and the
readers (``generate_report``, ``test_spectra``, ``cross_validate``) can never drift
apart. The layout is per-sector subfolders plus a ``summary/`` folder for global
artifacts:

    results/
      <sector_id>/            bb, cc, bc, cu
        params.csv
        errors.csv
        observables.csv
        <Wave>_Wave_GEM_Coefficients.csv
      summary/
        goodness_of_fit.csv
        consolidated_report.csv
        radiative_decays.csv
        cross_validation.csv
"""

import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))


def results_root():
    return _ROOT


def sector_dir(sector_id):
    """results/<sector_id>/ (created on demand)."""
    d = os.path.join(_ROOT, sector_id)
    os.makedirs(d, exist_ok=True)
    return d


def summary_dir():
    """results/summary/ (created on demand)."""
    d = os.path.join(_ROOT, "summary")
    os.makedirs(d, exist_ok=True)
    return d


def params_csv(sector_id):
    return os.path.join(sector_dir(sector_id), "params.csv")


def errors_csv(sector_id):
    return os.path.join(sector_dir(sector_id), "errors.csv")


def observables_csv(sector_id):
    return os.path.join(sector_dir(sector_id), "observables.csv")


def gem_csv(sector_id, wave):
    return os.path.join(sector_dir(sector_id), f"{wave}_Wave_GEM_Coefficients.csv")


def summary_csv(name):
    return os.path.join(summary_dir(), name)
