"""Load experimental quarkonium data straight from the official PDG SQLite dump.

This replaces the old hand-edited ``data/pdg_data.json``. The numbers now come live
from the PDG database (``data/pdg-*.sqlite``), so re-pointing at a newer edition is a
file swap, not a hand-merge.

The one thing that *can't* be read from the database is which measured particle is
which ``(n^{2S+1}L_J)`` state -- that mapping is physics, not data, so it lives here
in ``STATE_MAP`` (keyed by PDG identifier, e.g. ``M070`` = J/psi(1S)). Everything
else -- masses, total widths, e+e- and gamma-gamma partial widths -- is queried.

``load_pdg_data()`` returns the same dict shape the JSON used, so the rest of the
pipeline doesn't care where the numbers came from:

    {
      "bb": {"(1^3S)": 9.4603, ...},          # masses, GeV
      "cc": {...}, "bc": {...},
      "cc_widths_total_MeV": {...},
      "cc_widths_ee_keV": {...},
      "cc_widths_ee_err_keV": {...},          # experimental 1-sigma on the e+e- width
      "cc_widths_gammagamma_keV": {...},
      "cc_widths_gammagamma_err_keV": {...},  # experimental 1-sigma on the gamma-gamma width
      "bb_widths_*": {...},
    }
"""

import os
import glob
import sqlite3
from functools import lru_cache

# --- which PDG particle is which spectroscopic state -------------------------
# Keyed by PDG identifier (pdgid). None means "no PDG entry" -> emitted as null,
# matching the old JSON (the fitter skips null masses).
STATE_MAP = {
    "cc": {
        "(1^1S)": "M026",    # eta_c(1S)
        "(2^1S)": "M059",    # eta_c(2S)
        "(1^3S)": "M070",    # J/psi(1S)
        "(2^3S)": "M071",    # psi(2S)
        "(3^3S)": "M072",    # psi(4040)
        "(4^3S)": "M073",    # psi(4415)
        "(1^1P)": "M144",    # h_c(1P)
        "(1^3P_0)": "M056",  # chi_c0(1P)
        "(1^3P_1)": "M055",  # chi_c1(1P)
        "(1^3P_2)": "M057",  # chi_c2(1P)
        "(2^3P_2)": "M050",  # chi_c2(3930)
        "(1^3D_1)": "M053",  # psi(3770)
        "(2^3D_1)": "M025",  # psi(4160)
        "(1^3D_2)": "M212",  # psi_2(3823)
        "(1^3D_3)": "M241",  # psi_3(3842)
    },
    "bb": {
        "(1^1S)": "M171",    # eta_b(1S)
        "(2^1S)": "M200",    # eta_b(2S)
        "(1^3S)": "M049",    # Upsilon(1S)
        "(2^3S)": "M052",    # Upsilon(2S)
        "(3^3S)": "M048",    # Upsilon(3S)
        "(4^3S)": "M047",    # Upsilon(4S)
        "(1^1P)": "M204",    # h_b(1P)
        "(2^1P)": "M205",    # h_b(2P)
        "(1^3P_0)": "M076",  # chi_b0(1P)
        "(1^3P_1)": "M077",  # chi_b1(1P)
        "(1^3P_2)": "M078",  # chi_b2(1P)
        "(2^3P_0)": "M079",  # chi_b0(2P)
        "(2^3P_1)": "M080",  # chi_b1(2P)
        "(2^3P_2)": "M081",  # chi_b2(2P)
        "(1^3D_2)": "M177",  # Upsilon_2(1D)
    },
    "bc": {
        "(1^1S)": "S091",    # B_c+ ground state
        "(2^1S)": "M217",    # B_c(2S)+
        "(1^3S)": None,      # not yet observed
        "(2^3S)": None,      # not yet observed
    },
}

# Decay final states whose partial widths we benchmark against.
_EE = "e+ e-"
_GG = "gamma gamma"


def _is_vector(label):
    """J^PC = 1^-- (couples to e+e- via one photon): the 3S1 and 3D1 states."""
    return label.endswith("^3S)") or label.endswith("^3D_1)")


def _is_two_photon(label):
    """C = +, J != 1 (gamma-gamma allowed): 1S0 pseudoscalars, 3P0/3P2."""
    return (
        label.endswith("^1S)")
        or label.endswith("^3P_0)")
        or label.endswith("^3P_2)")
    )

# eV/keV/MeV/GeV -> keV
_TO_KEV = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3, "GeV": 1e6}
# eV/keV/MeV/GeV -> MeV
_TO_MEV = {"eV": 1e-6, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}


def default_db_path():
    """Newest ``data/pdg*.sqlite`` in the repo, so the filename can change freely."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    matches = sorted(glob.glob(os.path.join(root, "data", "pdg*.sqlite")))
    if not matches:
        raise FileNotFoundError(
            f"No PDG SQLite database found in {os.path.join(root, 'data')} "
            "(expected something like data/pdg-2026.0.sqlite)."
        )
    return matches[-1]


def _rows(cur, node_pdgid):
    """All numeric pdgdata rows for a node, freshest edition first."""
    return cur.execute(
        """SELECT value, error_positive, error_negative, unit_text, value_type,
                  in_summary_table, limit_type, edition
           FROM pdgdata WHERE pdgid = ? AND value IS NOT NULL
           ORDER BY edition DESC""",
        (node_pdgid,),
    ).fetchall()


def _pick(cur, node_pdgid):
    """The one representative value for a node: prefer the PDG summary/average,
    skip upper/lower limits. Returns the sqlite Row or None."""
    cand = [r for r in _rows(cur, node_pdgid) if r["limit_type"] is None]
    if not cand:
        return None
    for ok in (
        lambda r: r["in_summary_table"] == 1 and r["value_type"] == "AC",
        lambda r: r["in_summary_table"] == 1,
        lambda r: r["value_type"] == "AC",
        lambda r: r["value_type"] == "FC",
        lambda r: True,
    ):
        hit = [r for r in cand if ok(r)]
        if hit:
            return hit[0]
    return None


def _convert(value, unit_text, table):
    unit = (unit_text or "").strip()
    factor = table.get(unit)
    return value * factor if factor is not None else None


def _mass_gev(cur, pdgid):
    r = _pick(cur, pdgid + "M")
    if r is None:
        return None
    mev = _convert(r["value"], r["unit_text"], _TO_MEV)
    return mev / 1000.0 if mev is not None else None


def _mass_err_gev(cur, pdgid):
    """Symmetric experimental 1-sigma mass uncertainty in GeV (the larger of the
    asymmetric PDG errors), or None if unavailable."""
    r = _pick(cur, pdgid + "M")
    if r is None:
        return None
    ep = r["error_positive"]
    en = r["error_negative"]
    errs = [abs(e) for e in (ep, en) if e is not None]
    if not errs:
        return None
    mev = _convert(max(errs), r["unit_text"], _TO_MEV)
    return mev / 1000.0 if mev is not None else None


def _total_width_kev(cur, pdgid):
    r = _pick(cur, pdgid + "W")
    if r is None:
        return None
    return _convert(r["value"], r["unit_text"], _TO_KEV)


def _particle_name(cur, pdgid):
    r = cur.execute(
        "SELECT name FROM pdgparticle WHERE pdgid = ? ORDER BY mcid IS NULL, mcid DESC LIMIT 1",
        (pdgid,),
    ).fetchone()
    return r["name"] if r else None


def _row_err(r):
    """Symmetric experimental 1-sigma (the larger of the asymmetric PDG errors) in
    the row's own units, or None if neither error is recorded."""
    errs = [abs(e) for e in (r["error_positive"], r["error_negative"]) if e is not None]
    return max(errs) if errs else None


def _partial_width_kev(cur, pdgid, decay):
    """Partial width to ``decay`` (e.g. 'e+ e-') and its experimental 1-sigma error,
    both in keV: returns the tuple ``(width, err)`` (``(None, None)`` if absent;
    ``err`` may be ``None`` even when ``width`` is found).

    PDG stores this two ways: as a direct partial width on the
    ``G(<particle> --> <decay>)`` node (energy units), or only as a branching
    fraction (dimensionless) that we multiply by the total width. The unit on the
    value tells us which we're looking at. The experimental error rides along the
    same node (it is queried in :func:`_rows`); for the branching-fraction form the
    fractional error is scaled by the total width (the total's own error is a
    second-order term and is neglected).
    """
    pname = _particle_name(cur, pdgid)
    if pname is None:
        return None, None
    children = cur.execute(
        "SELECT pdgid, description FROM pdgid WHERE parent_pdgid = ?", (pdgid,)
    ).fetchall()

    # 1) direct partial-width node, e.g. "G(J/psi(1S) --> e+ e-)"
    direct = [c for c in children if c["description"] == f"G({pname} --> {decay})"]
    if direct:
        r = _pick(cur, direct[0]["pdgid"])
        if r is not None:
            width = _convert(r["value"], r["unit_text"], _TO_KEV)
            raw_err = _row_err(r)
            if width is not None:  # value carried energy units -> it's a width
                err = _convert(raw_err, r["unit_text"], _TO_KEV) if raw_err is not None else None
                return width, err
            total = _total_width_kev(cur, pdgid)  # dimensionless -> it's a fraction
            if total is None:
                return None, None
            return r["value"] * total, (raw_err * total if raw_err is not None else None)

    # 2) branching-fraction summary node, e.g. "chi_c2(1P) --> gamma gamma"
    bfx = [c for c in children if c["description"] == f"{pname} --> {decay}"]
    if bfx:
        r = _pick(cur, bfx[0]["pdgid"])
        if r is not None:
            total = _total_width_kev(cur, pdgid)
            if total is None:
                return None, None
            raw_err = _row_err(r)
            return r["value"] * total, (raw_err * total if raw_err is not None else None)
    return None, None


@lru_cache(maxsize=None)
def load_pdg_data(db_path=None):
    """Build the experimental-data dict (masses GeV, widths) from the PDG SQLite.

    Same shape as the retired ``data/pdg_data.json``. Cached per ``db_path``.
    """
    db_path = db_path or default_db_path()
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    try:
        data = {}
        for sector, states in STATE_MAP.items():
            masses, mass_err, total = {}, {}, {}
            ee, ee_err, gg, gg_err = {}, {}, {}, {}
            for label, pdgid in states.items():
                if pdgid is None:
                    masses[label] = None
                    continue
                masses[label] = _mass_gev(cur, pdgid)
                merr = _mass_err_gev(cur, pdgid)
                if merr is not None:
                    mass_err[label] = merr

                tw = _total_width_kev(cur, pdgid)
                if tw is not None:
                    total[label] = tw / 1000.0  # keV -> MeV

                if _is_vector(label):
                    w_ee, w_ee_err = _partial_width_kev(cur, pdgid, _EE)
                    if w_ee is not None:
                        ee[label] = w_ee
                        if w_ee_err is not None:
                            ee_err[label] = w_ee_err
                if _is_two_photon(label):
                    w_gg, w_gg_err = _partial_width_kev(cur, pdgid, _GG)
                    if w_gg is not None:
                        gg[label] = w_gg
                        if w_gg_err is not None:
                            gg_err[label] = w_gg_err

            data[sector] = masses
            data[f"{sector}_mass_err_GeV"] = mass_err
            data[f"{sector}_widths_total_MeV"] = total
            data[f"{sector}_widths_ee_keV"] = ee
            data[f"{sector}_widths_ee_err_keV"] = ee_err
            data[f"{sector}_widths_gammagamma_keV"] = gg
            data[f"{sector}_widths_gammagamma_err_keV"] = gg_err
        return data
    finally:
        con.close()
