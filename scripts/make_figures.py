#!/usr/bin/env python3
r"""Generate every analysis figure for the heavy-meson GEM project.

Reads only the CSVs already written under ``results/`` (so it is fast and does
*not* re-run the slow fit) plus the curated literature values in
``literature_data.py``. Figures are written to ``figures/`` as both vector PDF
(for the LaTeX paper) and PNG (for quick viewing).

Run:

    conda run -n 400 python scripts/make_figures.py

Figure groups
-------------
A. GEM basis / wavefunctions (the "expansion" plots):
   - per-state Gaussian coverage (individual weighted basis fns + total)
   - reconstructed radial amplitudes u(r)=rR(r) showing radial nodes
   - the geometric Gaussian width mesh
B. Mass spectra & Hamiltonian-vs-literature comparison (with computation
   error bars):
   - Grotrian-style level diagrams (calc vs PDG)
   - this work vs BGS-2005 (NR & GI) for charmonium
   - this work vs Godfrey-Moats-2015 (GI) for bottomonium
C. Fit quality: mass residuals with systematic band, pulls, chi2/dof, RMS,
   cross-validation.
D. Decays & static observables: leptonic, two-photon, radiative E1/M1 widths,
   RMS radii, |R(0)|^2.
E. Physics inputs: Cornell potential per sector, running-coupling
   interpolation used for B_c.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import gamma

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from quarkonia import paths  # noqa: E402

sys.path.append(os.path.dirname(__file__))
import literature_data as lit  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths & global style
# --------------------------------------------------------------------------- #
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
    }
)

# Constituent quark masses (GeV) and fitted sectors -- mirror run_spectrum.py
SECTORS = {
    "bb": {"label": r"Bottomonium $b\bar b$", "m1": 4.730, "m2": 4.730, "color": "#1f4e8c"},
    "cc": {"label": r"Charmonium $c\bar c$", "m1": 1.500, "m2": 1.500, "color": "#b22222"},
    "bc": {"label": r"$B_c$ $(b\bar c)$", "m1": 4.730, "m2": 1.500, "color": "#2e8b57"},
    "cu": {"label": r"$D$ $(c\bar u)$", "m1": 1.500, "m2": 0.330, "color": "#8030a0"},
}
HBARC_FM = 0.197327  # GeV*fm  (1 GeV^-1 = 0.197327 fm)

# Experimental data are loaded from data/pdg-2026.0.sqlite -> PDG 2026 edition.
PDG_LABEL = "PDG 2026"

L_LETTER = {0: "S", 1: "P", 2: "D", 3: "F"}
LETTER_L = {v: k for k, v in L_LETTER.items()}

_saved = []


def save(fig, name):
    """Save a figure as both PDF and PNG and record it."""
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    _saved.append(name)
    print(f"  saved figures/{name}.pdf / .png")


# --------------------------------------------------------------------------- #
# State-name parsing
# --------------------------------------------------------------------------- #
_STATE_RE = re.compile(r"\((\d)\^(\d)([SPDF])(?:_(\d))?\)")


def parse_state(name):
    """('(2^3P_1) chi_c') -> dict(n, mult, S, l, j, letter) or None."""
    m = _STATE_RE.search(str(name))
    if not m:
        return None
    n = int(m.group(1))
    mult = int(m.group(2))  # 2S+1
    letter = m.group(3)
    l = LETTER_L[letter]
    if m.group(4) is not None:
        j = int(m.group(4))
    else:
        j = (1 if mult == 3 else 0) if l == 0 else l
    return {"n": n, "mult": mult, "S": (mult - 1) // 2, "l": l, "j": j, "letter": letter}


def canonical_key(name):
    """Spectroscopic key 'n 2S+1 L J', e.g. '1 3S1', matching literature_data."""
    p = parse_state(name)
    if p is None:
        return None
    return f"{p['n']} {p['mult']}{p['letter']}{p['j']}"


def jpc(p):
    """J^{PC} string for a (l, S, j) state (qqbar): P=(-1)^{L+1}, C=(-1)^{L+S}."""
    P = "-" if (p["l"] + 1) % 2 else "+"
    C = "-" if (p["l"] + p["S"]) % 2 else "+"
    return f"{p['j']}^{{{P}{C}}}"


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
def load_params():
    out = {}
    for sid in SECTORS:
        df = pd.read_csv(paths.params_csv(sid))
        out[sid] = df.iloc[0].to_dict()
    return out


def load_consolidated():
    df = pd.read_csv(paths.summary_csv("consolidated_report.csv"))
    return df


def load_masses(df, sector):
    """Return DataFrame of mass rows for a sector with parsed quantum numbers."""
    sub = df[(df["Sector"] == sector.upper()) & (df["Property"] == "Mass (GeV)")].copy()
    sub["key"] = sub["State"].apply(canonical_key)
    sub["parsed"] = sub["State"].apply(parse_state)
    return sub


def load_gem(sid, wave):
    path = paths.gem_csv(sid, wave)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# GEM wavefunction reconstruction (matches gem_solver.solve_gem conventions)
# --------------------------------------------------------------------------- #
def basis_norms(nu, l):
    r"""norms_i = sqrt( 1/2 * Gamma(l+3/2) / (2 nu_i)^{l+3/2} )."""
    return np.sqrt(0.5 * gamma(l + 1.5) / (2.0 * nu) ** (l + 1.5))


def reconstruct_u(r, nu, coeffs, l):
    r"""Reduced radial amplitude u(r)=r R(r) = sum_i c_i r^{l+1} e^{-nu_i r^2}/norm_i.

    Normalised so that \int_0^\infty u(r)^2 dr = 1 (inherited from the solver)."""
    norms = basis_norms(nu, l)
    u = np.zeros_like(r)
    for ci, nui, ni in zip(coeffs, nu, norms):
        u += ci * r ** (l + 1) * np.exp(-nui * r**2) / ni
    return u


def r0_density(nu, coeffs):
    r"""|R(0)|^2 for an S-wave: R(0)=sum_i c_i/norm_i (GeV^3)."""
    norms = basis_norms(nu, 0)
    return float(np.sum(coeffs / norms)) ** 2


# Map sector -> S-wave column labels present in the coefficient CSVs.
S_STATE_COLS = {
    "bb": [("c_1S", "1S"), ("c_2S", "2S"), ("c_3S", "3S")],
    "cc": [("c_1S", "1S"), ("c_2S", "2S"), ("c_3S", "3S")],
    "bc": [("c_1S", "1S"), ("c_2S", "2S"), ("c_3S", "3S")],
    "cu": [("c_1S", "1S")],
}


# =========================================================================== #
# GROUP A -- GEM basis / wavefunction "expansion" plots
# =========================================================================== #
def plot_gem_coverage_single(sid="bb", col="c_1S", label="1S"):
    """Paper-ready single coverage plot: weighted Gaussians + total wavefn."""
    df = load_gem(sid, "S")
    if df is None or col not in df:
        return
    nu = df["nu_width"].values
    coeffs = df[col].values
    norms = basis_norms(nu, 0)
    r = np.logspace(-2, 1.4, 2000)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(nu)))
    total = np.zeros_like(r)
    for ci, nui, ni, cl in zip(coeffs, nu, norms, colors):
        u_i = ci * r * np.exp(-nui * r**2) / ni
        total += u_i
        ax.plot(r, u_i, "--", color=cl, alpha=0.55, lw=1.0)
    ax.plot(r, total, color="black", lw=2.6,
            label=rf"Total $u(r)=rR(r)$, {SECTORS[sid]['label']} {label}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 30)
    ax.set_xlabel(r"$r$  [GeV$^{-1}$]")
    ax.set_ylabel(r"Radial amplitude $u(r)=rR(r)$")
    ax.set_title(f"GEM basis coverage: {SECTORS[sid]['label']} {label}")
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label(r"basis index $n$ (narrow $\to$ wide)")
    ax.legend(loc="upper right")
    save(fig, f"A1_gem_coverage_{sid}_{label}")


def plot_gem_coverage_grid():
    """2x3 grid of coverage plots for bb and cc {1S,2S,3S}."""
    r = np.logspace(-2, 1.4, 1500)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for row, sid in enumerate(("bb", "cc")):
        df = load_gem(sid, "S")
        if df is None:
            continue
        nu = df["nu_width"].values
        norms = basis_norms(nu, 0)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(nu)))
        for cidx, (col, lbl) in enumerate(S_STATE_COLS[sid]):
            ax = axes[row, cidx]
            coeffs = df[col].values
            total = np.zeros_like(r)
            for ci, nui, ni, cl in zip(coeffs, nu, norms, colors):
                u_i = ci * r * np.exp(-nui * r**2) / ni
                total += u_i
                ax.plot(r, u_i, "--", color=cl, alpha=0.45, lw=0.8)
            ax.plot(r, total, color="black", lw=2.2)
            ax.axhline(0, color="0.5", lw=0.7)
            ax.set_xscale("log")
            ax.set_xlim(1e-2, 30)
            ax.set_title(f"{SECTORS[sid]['label']} {lbl}", fontsize=11)
            if cidx == 0:
                ax.set_ylabel(r"$u(r)=rR(r)$")
            if row == 1:
                ax.set_xlabel(r"$r$  [GeV$^{-1}$]")
    fig.suptitle("GEM basis-function coverage of the radial wavefunctions "
                 "(dashed: weighted Gaussians, solid: total)", y=0.98)
    save(fig, "A2_gem_coverage_grid")


def plot_radial_wavefunctions():
    """u(r) for 1S/2S/3S per sector, showing radial nodes."""
    r = np.linspace(1e-3, 12, 2000)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, sid in zip(axes, ("bb", "cc", "bc")):
        df = load_gem(sid, "S")
        if df is None:
            continue
        nu = df["nu_width"].values
        for (col, lbl), c in zip(S_STATE_COLS[sid],
                                 ["#1f4e8c", "#b22222", "#2e8b57"]):
            if col not in df:
                continue
            u = reconstruct_u(r, nu, df[col].values, 0)
            # fix overall sign so the inner lobe is positive
            if u[np.argmax(np.abs(u))] < 0:
                u = -u
            ax.plot(r * HBARC_FM, u, color=c, label=lbl)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_title(SECTORS[sid]["label"])
        ax.set_xlabel(r"$r$  [fm]")
        ax.set_xlim(0, 12 * HBARC_FM)
        ax.legend(title="state")
    axes[0].set_ylabel(r"$u(r)=rR(r)$  (normalised)")
    fig.suptitle("Reconstructed S-wave radial amplitudes (radial nodes visible)", y=1.0)
    save(fig, "A3_radial_wavefunctions")


def plot_gaussian_mesh():
    """Geometric width mesh nu_n and the corresponding ranges r_n=1/sqrt(nu_n)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    sid = "bb"
    df = load_gem(sid, "S")
    nu = df["nu_width"].values
    n = np.arange(1, len(nu) + 1)
    ax1.semilogy(n, nu, "o-", color=SECTORS[sid]["color"])
    ax1.set_xlabel(r"basis index $n$")
    ax1.set_ylabel(r"$\nu_n$  [GeV$^2$]")
    ax1.set_title(r"Geometric Gaussian widths $\nu_n=1/r_n^2$")

    r_n = 1.0 / np.sqrt(nu)
    ax2.semilogy(n, r_n * HBARC_FM, "s-", color="#b8860b")
    ax2.set_xlabel(r"basis index $n$")
    ax2.set_ylabel(r"range $r_n=1/\sqrt{\nu_n}$  [fm]")
    ax2.set_title("Length scales spanned by the basis")
    ax2.axhspan(0.1, 1.0, color="green", alpha=0.08)
    ax2.text(0.5, 0.4, "typical meson size", transform=ax2.transAxes,
             fontsize=9, color="green")
    fig.suptitle(r"GEM geometric mesh ($n_{\max}=25$, $r_{\min}=0.05$, "
                 r"$r_{\max}=15$ GeV$^{-1}$)", y=1.02)
    save(fig, "A4_gaussian_mesh")


# =========================================================================== #
# GROUP B -- Mass spectra & literature comparison
# =========================================================================== #
def _spectrum_panel(ax, sub, title):
    """Grotrian-style level diagram: calc (with model err) vs PDG, columns by L."""
    handles_done = False
    for _, row in sub.iterrows():
        p = row["parsed"]
        if p is None:
            continue
        x = p["l"] + 0.0
        calc = row["Calculated"]
        unc = row["Uncertainty"] / 1000.0  # MeV -> GeV
        # small horizontal jitter so degenerate-L states don't overlap
        jit = (p["j"] - p["l"]) * 0.06 + (0.04 if p["mult"] == 1 else -0.04)
        xc = x + jit
        ax.errorbar(xc, calc, yerr=unc, fmt="_", ms=18, color="#1f4e8c",
                    elinewidth=1.0, capsize=2,
                    label=("This work (calc.)" if not handles_done else None))
        if pd.notna(row["Experimental"]):
            ax.plot(xc, row["Experimental"], "_", ms=18, color="#b22222",
                    label=(PDG_LABEL if not handles_done else None))
        handles_done = True
    ax.set_xticks(range(0, 3))
    ax.set_xticklabels(["S", "P", "D"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel("orbital angular momentum $L$")
    ax.set_ylabel("Mass [GeV]")
    ax.set_title(title)
    ax.legend(loc="lower right")


def plot_spectrum(sector_id):
    df = load_consolidated()
    sub = load_masses(df, sector_id)
    fig, ax = plt.subplots(figsize=(7, 6))
    _spectrum_panel(ax, sub, f"{SECTORS[sector_id]['label']} spectrum "
                             "(bars: model uncertainty)")
    save(fig, f"B_spectrum_{sector_id}")


def plot_literature_comparison(sector_id, models):
    """Per-state deviation from experiment, M_model - M_exp [MeV].

    ``models`` is ``{model_label: {state_key: mass_MeV}}``. This work is overlaid
    with its model-uncertainty error bars; every literature model is a marker."""
    df = load_consolidated()
    sub = load_masses(df, sector_id)
    sub = sub[pd.notna(sub["Experimental"])].copy()

    rows = []
    for _, row in sub.iterrows():
        key = row["key"]
        if not any(key in mv for mv in models.values()):
            continue
        exp = row["Experimental"] * 1000.0
        p = row["parsed"]
        rec = {
            "label": row["State"].strip(),
            "sortkey": (p["l"], p["n"], -p["j"]),
            "This work": row["Calculated"] * 1000.0 - exp,
            "This work_err": row["Uncertainty"],
        }
        for ml, mv in models.items():
            if key in mv:
                rec[ml] = mv[key] - exp
        rows.append(rec)
    rows.sort(key=lambda d: d["sortkey"])
    if not rows:
        return

    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))
    all_models = ["This work"] + list(models.keys())
    offsets = np.linspace(-0.32, 0.32, len(all_models))

    fig, ax = plt.subplots(figsize=(10.5, 0.66 * len(rows) + 2.0))
    ax.axvspan(-10, 10, color="green", alpha=0.07, zorder=0,
               label=r"$\pm10$ MeV (theory syst.)")
    ax.axvline(0, color="black", lw=1.0, zorder=1)
    for off, ml in zip(offsets, all_models):
        color, marker = lit.MODEL_STYLE.get(ml, ("gray", "o"))
        xs = [r.get(ml, np.nan) for r in rows]
        if ml == "This work":
            ax.errorbar(xs, y + off, xerr=[r["This work_err"] for r in rows],
                        fmt=marker, color=color, ms=7, capsize=2, elinewidth=1.0,
                        mec="black", mew=0.6, label=ml, zorder=6)
        else:
            ax.plot(xs, y + off, marker, color=color, ms=6.5, alpha=0.9,
                    mec="black", mew=0.3, label=ml, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$M_{\rm model}-M_{\rm exp}$  [MeV]")
    ax.set_title(f"{SECTORS[sector_id]['label']}: Hamiltonian vs. literature\n"
                 "(this work's bars = model uncertainty; "
                 "literature values are predictions)")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.invert_yaxis()
    ax.margins(y=0.012)
    save(fig, f"B_literature_{sector_id}")


def plot_model_rms_comparison():
    """Aggregate accuracy: RMS(M_model - M_exp) per model, over a common state set."""
    df = load_consolidated()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, sector_id, models in ((axes[0], "cc", lit.CHARMONIUM),
                                  (axes[1], "bb", lit.BOTTOMONIUM)):
        sub = load_masses(df, sector_id)
        sub = sub[pd.notna(sub["Experimental"])].copy()
        exp = {r["key"]: r["Experimental"] * 1000.0
               for _, r in sub.iterrows() if r["key"]}
        thiswork = {r["key"]: r["Calculated"] * 1000.0
                    for _, r in sub.iterrows() if r["key"]}
        # fair comparison: only states present in experiment, this work, AND every model
        common = set(exp) & set(thiswork)
        for mv in models.values():
            common &= set(mv)
        common = sorted(common)

        def rms(masses):
            return float(np.sqrt(np.mean([(masses[k] - exp[k]) ** 2 for k in common])))

        entries = [("This work", rms(thiswork))]
        entries += [(ml, rms(mv)) for ml, mv in models.items()]
        entries.sort(key=lambda e: e[1])
        labels = [e[0] for e in entries]
        vals = [e[1] for e in entries]
        colors = [lit.MODEL_STYLE.get(l, ("gray",))[0] for l in labels]
        xb = np.arange(len(entries))
        bars = ax.bar(xb, vals, color=colors, alpha=0.9)
        # mark "This work" (fitted, not predicted) distinctly
        for lab, bar in zip(labels, bars):
            if lab == "This work":
                bar.set_hatch("//")
                bar.set_edgecolor("black")
        for xi, v in zip(xb, vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xb)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(r"RMS$(M_{\rm model}-M_{\rm exp})$  [MeV]")
        ax.set_title(f"{SECTORS[sector_id]['label']}  "
                     f"($N={len(common)}$ common states)")
    fig.suptitle("Aggregate accuracy per model "
                 "(this work, hatched, is fitted to these masses; others predict)",
                 y=1.02)
    save(fig, "B_model_rms")


# =========================================================================== #
# GROUP C -- Fit-quality diagnostics
# =========================================================================== #
def plot_mass_residuals():
    df = load_consolidated()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, sid in zip(axes, ("bb", "cc")):
        sub = load_masses(df, sid)
        sub = sub[pd.notna(sub["Experimental"])].copy()
        sub = sub.sort_values("Experimental")
        x = np.arange(len(sub))
        res = (sub["Calculated"] - sub["Experimental"]) * 1000.0
        err = sub["Uncertainty"].values
        ax.bar(x, res, color=SECTORS[sid]["color"], alpha=0.55, width=0.6)
        ax.errorbar(x, res, yerr=err, fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=2)
        ax.axhspan(-10, 10, color="green", alpha=0.10,
                   label=r"$\pm10$ MeV theory syst.")
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([s.strip() for s in sub["State"]],
                           rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(r"$M_{\rm calc}-M_{\rm exp}$  [MeV]")
        ax.set_title(f"{SECTORS[sid]['label']} mass residuals "
                     "(bars: model uncertainty)")
        ax.legend(loc="upper right")
    save(fig, "C1_mass_residuals")


def plot_pulls():
    df = load_consolidated()
    fig, ax = plt.subplots(figsize=(11, 5))
    xbase = 0
    ticks, ticklabels = [], []
    for sid in ("bb", "cc"):
        sub = load_masses(df, sid)
        sub = sub[pd.notna(sub["Pull_sigma"])].copy().sort_values("Experimental")
        x = np.arange(len(sub)) + xbase
        ax.bar(x, sub["Pull_sigma"], color=SECTORS[sid]["color"], alpha=0.7,
               width=0.7, label=SECTORS[sid]["label"])
        ticks += list(x)
        ticklabels += [s.strip() for s in sub["State"]]
        xbase = x[-1] + 2
    for k in (-1, 1):
        ax.axhline(k, color="gray", ls="--", lw=1)
    for k in (-2, 2):
        ax.axhline(k, color="gray", ls=":", lw=0.8)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(r"pull $=(M_{\rm calc}-M_{\rm exp})/\sigma_{\rm phys}$")
    ax.set_title(r"Mass pulls vs. physical uncertainty "
                 r"($\sigma_{\rm phys}=\sigma_{\rm exp}\oplus10$ MeV)")
    ax.legend()
    save(fig, "C2_pulls")


def plot_chi2_rms():
    gof = pd.read_csv(paths.summary_csv("goodness_of_fit.csv"))
    cv = pd.read_csv(paths.summary_csv("cross_validation.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    # chi2/dof
    g = gof[gof["chi2_per_dof"].notna()]
    axes[0].bar(g["sector_id"], g["chi2_per_dof"],
                color=[SECTORS[s]["color"] for s in g["sector_id"]])
    axes[0].axhline(1.0, color="black", ls="--", lw=1, label=r"$\chi^2/{\rm dof}=1$")
    axes[0].set_ylabel(r"$\chi^2/{\rm dof}$")
    axes[0].set_title("Reduced chi-square (mass + width fit)")
    axes[0].legend()

    # RMS deviation
    axes[1].bar(gof["sector_id"], gof["rms_mev"],
                color=[SECTORS[s]["color"] for s in gof["sector_id"]])
    axes[1].set_ylabel("RMS mass deviation [MeV]")
    axes[1].set_title("Model-independent RMS deviation")
    axes[1].set_yscale("log")

    # cross-validation
    sids = []
    for nm in cv["sector"]:
        sids.append("bb" if "b_bbar" in nm else "cc")
    xp = np.arange(len(cv))
    w = 0.38
    axes[2].bar(xp - w / 2, cv["full_rms_mev"], w, label="train (full fit)",
                color="#4c72b0")
    axes[2].bar(xp + w / 2, cv["loo_rms_mev"], w, label="leave-one-out (test)",
                color="#dd8452")
    axes[2].set_xticks(xp)
    axes[2].set_xticklabels(sids)
    axes[2].set_ylabel("RMS [MeV]")
    axes[2].set_title("Cross-validation (overfitting check)")
    axes[2].legend()
    save(fig, "C3_chi2_rms_crossval")


# =========================================================================== #
# GROUP D -- Decay widths & static observables
# =========================================================================== #
def _width_rows(df, sector, kind):
    sub = df[(df["Sector"] == sector.upper()) & (df["Property"] == kind)].copy()
    sub = sub[pd.notna(sub["Experimental"])]
    return sub


def plot_width_comparison(kind, fname, title):
    df = load_consolidated()
    rows = []
    tag = {"cc": r"$c\bar c$", "bb": r"$b\bar b$"}
    for sid in ("cc", "bb"):
        for _, row in _width_rows(df, sid, kind).iterrows():
            rows.append((f"{tag[sid]} {row['State'].strip()}",
                         row["Calculated"], row["Uncertainty"], row["Experimental"]))
    if not rows:
        return
    labels, calc, err, exp = zip(*rows)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(rows) + 1.6))
    w = 0.38
    ax.barh(y + w / 2, calc, w, xerr=err, color="#1f4e8c", alpha=0.85,
            label="This work", capsize=2, error_kw={"elinewidth": 1})
    ax.barh(y - w / 2, exp, w, color="#b22222", alpha=0.85, label=PDG_LABEL)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$\Gamma$  [keV]")
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
    save(fig, fname)


def plot_radiative_widths():
    rad = pd.read_csv(paths.summary_csv("radiative_decays.csv"))
    # Experimental reference values (keV); aligned with the user's paper
    # Table (tab:radiative_decays). None where no firm measurement exists.
    exp_map = {
        "(1^3S) J/ψ -> (1^1S) η_c + γ": 1.58,
        "(2^3S) ψ -> (2^1S) η_c + γ": 0.05,
        "(1^3P_0) χ_c -> (1^3S) J/ψ + γ": 116.0,
        "(1^3P_1) χ_c -> (1^3S) J/ψ + γ": 288.0,
        "(1^3S) Υ_b -> (1^1S) η_b + γ": None,
        "(2^3S) Υ -> (2^1S) η_b + γ": None,
        "(1^3P_0) χ_b -> (1^3S) Υ_b + γ": 27.5,
        "(1^3P_1) χ_b -> (1^3S) Υ_b + γ": 31.1,
    }
    rows = []
    for _, r in rad.iterrows():
        exp = exp_map.get(r["Transition"], None)
        short = (r["Transition"].replace("(1^3S)", "").replace("(1^1S)", "")
                 .replace(" + γ", "γ"))
        rows.append((f"[{r['Type']}] {r['Transition']}", r["Width_keV"],
                     r["Width_err_keV"], exp))
    labels, calc, err, exp = zip(*rows)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(rows) + 1.6))
    w = 0.38
    ax.barh(y + w / 2, calc, w, xerr=err, color="#2e8b57", alpha=0.85,
            label="This work", capsize=2, error_kw={"elinewidth": 1})
    exp_y = [(yy - w / 2) for yy, e in zip(y, exp) if e is not None]
    exp_v = [e for e in exp if e is not None]
    ax.barh(exp_y, exp_v, w, color="#b22222", alpha=0.85, label="Experiment")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Gamma$  [keV]  (log scale)")
    ax.set_title("Radiative E1/M1 transition widths vs. experiment")
    ax.legend()
    ax.invert_yaxis()
    save(fig, "D3_radiative_widths")


def plot_rms_radii():
    fig, ax = plt.subplots(figsize=(9.5, 5))
    markers = {"bb": "o", "cc": "s", "bc": "^"}
    for sid in ("bb", "cc", "bc"):
        path = paths.observables_csv(sid)
        if not os.path.exists(path):
            continue
        o = pd.read_csv(path)
        rms = o[o["Observable_Type"].str.contains("RMS")].copy()
        labels, vals = [], []
        for _, r in rms.iterrows():
            p = parse_state(r["State"])
            if p is None:
                continue
            labels.append(f"{p['n']}{p['letter']}")
            vals.append(r["Value_keV"] * HBARC_FM)  # GeV^-1 -> fm
        # keep first occurrence per (n,L) label
        seen = {}
        for lab, v in zip(labels, vals):
            seen.setdefault(lab, v)
        labs = list(seen.keys())
        xs = np.arange(len(labs))
        ax.plot(xs, [seen[k] for k in labs], markers[sid] + "-",
                color=SECTORS[sid]["color"], label=SECTORS[sid]["label"])
        if sid == "bb":
            ax.set_xticks(xs)
            ax.set_xticklabels(labs)
    ax.set_xlabel("state $nL$")
    ax.set_ylabel(r"RMS radius $\sqrt{\langle r^2\rangle}$  [fm]")
    ax.set_title("RMS radii: tighter binding for heavier systems")
    ax.legend()
    save(fig, "D4_rms_radii")


def plot_wavefunction_at_origin():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.35
    sectors = ["bb", "cc", "bc"]
    allkeys = ["1S", "2S", "3S"]
    for i, sid in enumerate(sectors):
        df = load_gem(sid, "S")
        if df is None:
            continue
        nu = df["nu_width"].values
        vals = []
        for col, lbl in S_STATE_COLS[sid]:
            vals.append(r0_density(nu, df[col].values) if col in df else np.nan)
        # pad to 3
        vals = vals + [np.nan] * (3 - len(vals))
        x = np.arange(len(allkeys)) + (i - 1) * width
        ax.bar(x, vals, width, color=SECTORS[sid]["color"],
               label=SECTORS[sid]["label"])
    ax.set_xticks(np.arange(len(allkeys)))
    ax.set_xticklabels(allkeys)
    ax.set_ylabel(r"$|R(0)|^2$  [GeV$^3$]")
    ax.set_xlabel("S-wave state")
    ax.set_title(r"Wavefunction at the origin $|R(0)|^2$ "
                 "(drives annihilation widths)")
    ax.legend()
    save(fig, "D5_wavefunction_origin")


# =========================================================================== #
# GROUP E -- Physics inputs
# =========================================================================== #
def plot_cornell_potentials(params):
    r = np.linspace(0.04, 3.0, 600)  # GeV^-1
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for sid, pr in params.items():
        V = -4.0 / 3.0 * pr["alpha_s"] / r + pr["b"] * r + pr["c"]
        ax.plot(r * HBARC_FM, V, color=SECTORS[sid]["color"],
                label=f"{SECTORS[sid]['label']} "
                      rf"($\alpha_s$={pr['alpha_s']:.2f}, $b$={pr['b']:.2f})")
    ax.set_xlabel(r"$r$  [fm]")
    ax.set_ylabel(r"$V_{\rm Cornell}(r)=-\frac{4}{3}\frac{\alpha_s}{r}+br+c$  [GeV]")
    ax.set_title("Fitted Cornell potentials by sector")
    ax.set_ylim(-1.6, 2.2)
    ax.legend()
    save(fig, "E1_cornell_potentials")


def plot_running_coupling(params):
    """alpha_s vs reduced-mass scale; show the log interpolation used for B_c."""
    mu = {sid: SECTORS[sid]["m1"] * SECTORS[sid]["m2"]
              / (SECTORS[sid]["m1"] + SECTORS[sid]["m2"]) for sid in SECTORS}
    a_cc, a_bb = params["cc"]["alpha_s"], params["bb"]["alpha_s"]
    mu_cc, mu_bb = mu["cc"], mu["bb"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    # interpolation curve in 1/alpha vs ln(mu)
    mu_line = np.linspace(min(mu.values()) * 0.9, max(mu.values()) * 1.1, 200)
    inv_line = (1.0 / a_cc + (np.log(mu_line / mu_cc) / np.log(mu_bb / mu_cc))
                * (1.0 / a_bb - 1.0 / a_cc))
    ax1.plot(mu_line, 1.0 / inv_line, "-", color="gray",
             label=r"RG log-interp. (anchors $c\bar c$, $b\bar b$)")
    for sid in SECTORS:
        fitted = sid in ("cc", "bb")
        ax1.plot(mu[sid], params[sid]["alpha_s"],
                 "o" if fitted else "D", ms=11, color=SECTORS[sid]["color"],
                 label=f"{SECTORS[sid]['label']} "
                       + ("(fitted)" if fitted else "(interp./frozen)"))
    ax1.set_xlabel(r"reduced mass $\mu=m_1 m_2/(m_1+m_2)$  [GeV]")
    ax1.set_ylabel(r"$\alpha_s$")
    ax1.set_title(r"Effective coupling vs. mass scale")
    ax1.legend(fontsize=8)

    # string tension b and offset c per sector (ordered by mu for a clean line)
    sids = sorted(SECTORS, key=lambda s: mu[s])
    ax2.plot([mu[s] for s in sids], [params[s]["b"] for s in sids], "o-",
             color="#b8860b", label=r"string tension $b$ [GeV$^2$]")
    ax2.plot([mu[s] for s in sids], [params[s]["c"] for s in sids], "s--",
             color="#555555", label=r"offset $c$ [GeV]")
    for s in sids:
        ax2.annotate(s, (mu[s], params[s]["b"]), fontsize=8,
                     textcoords="offset points", xytext=(4, 4))
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_xlabel(r"reduced mass $\mu$  [GeV]")
    ax2.set_ylabel("parameter value")
    ax2.set_title("Other fitted Cornell parameters")
    ax2.legend(fontsize=8)
    save(fig, "E2_running_coupling")


# Short sector tags for compact labels.
SECTOR_TAG = {"bb": r"$b\bar b$", "cc": r"$c\bar c$",
              "bc": r"$b\bar c$", "cu": r"$c\bar u$"}


def plot_cornell_literature(params):
    """Cornell potential vs. literature, plus the accepted string tension.

    Left: the spin-independent Cornell *shape* V(r)=-(4/3)alpha_s/r + b r (the
    unphysical constant c is dropped) for this work's sectors and the canonical
    Eichten et al. Cornell potential. Right: the string tension b across this
    work and the literature, against the accepted lattice/Regge value
    sigma ~ 0.18 GeV^2.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- Left: V(r) shapes vs. the canonical Cornell potential ----
    r = np.linspace(0.04, 3.2, 600)  # GeV^-1
    for sid, pr in params.items():
        Vhat = -4.0 / 3.0 * pr["alpha_s"] / r + pr["b"] * r
        axL.plot(r * HBARC_FM, Vhat, color=SECTORS[sid]["color"], lw=2.0,
                 label=f"This work {SECTOR_TAG[sid]}  "
                       rf"($\alpha_s$={pr['alpha_s']:.2f}, $b$={pr['b']:.3f})")
    cp = lit.CORNELL_PARAMS["Cornell (Eichten 1980)"]
    axL.plot(r * HBARC_FM, -4.0 / 3.0 * cp["alpha_s"] / r + cp["b"] * r, "k--",
             lw=2.2, label=r"Cornell/Eichten 1980 ($\alpha_s$=0.39, $b$=0.183)")
    axL.axhline(0, color="0.6", lw=0.7)
    axL.set_xlabel(r"$r$  [fm]")
    axL.set_ylabel(r"$\hat V(r)=-\frac{4}{3}\,\frac{\alpha_s}{r}+b\,r$  [GeV]")
    axL.set_title("Cornell potential shape vs. the canonical form\n"
                  "(unphysical constant offset $c$ removed)")
    axL.set_ylim(-1.6, 2.2)
    axL.set_xlim(0, 3.2 * HBARC_FM)
    axL.legend(fontsize=8, loc="lower right")

    # ---- Right: string tension b vs. literature & accepted value ----
    entries = [(f"This work {SECTOR_TAG[s]}", params[s]["b"], SECTORS[s]["color"])
               for s in ("bb", "cc", "bc", "cu")]
    entries += [(lab, d["b"], "0.55") for lab, d in lit.CORNELL_PARAMS.items()]
    labels, vals, colors = zip(*entries)
    xb = np.arange(len(entries))
    axR.bar(xb, vals, color=colors, edgecolor="black", linewidth=0.4)
    s0, ds = lit.ACCEPTED_STRING_TENSION
    axR.axhspan(s0 - ds, s0 + ds, color="green", alpha=0.13,
                label=rf"accepted $\sigma={s0}\pm{ds}$ GeV$^2$ (lattice/Regge)")
    axR.axhline(s0, color="green", lw=1.0, ls="--")
    for xi, v in zip(xb, vals):
        axR.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    axR.set_xticks(xb)
    axR.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axR.set_ylabel(r"string tension $b$  [GeV$^2$]")
    axR.set_ylim(0, max(vals) * 1.18)
    axR.set_title("String tension vs. literature & accepted value")
    axR.legend(fontsize=8, loc="upper left")
    save(fig, "E3_cornell_literature")


# =========================================================================== #
def main():
    print(f"Writing figures to {FIG_DIR}\n")
    params = load_params()

    print("Group A: GEM basis / wavefunctions")
    plot_gem_coverage_single("bb", "c_1S", "1S")
    plot_gem_coverage_single("cc", "c_1S", "1S")
    plot_gem_coverage_grid()
    plot_radial_wavefunctions()
    plot_gaussian_mesh()

    print("Group B: mass spectra & literature comparison")
    for sid in ("cc", "bb", "bc"):
        plot_spectrum(sid)
    plot_literature_comparison("cc", lit.CHARMONIUM)
    plot_literature_comparison("bb", lit.BOTTOMONIUM)
    plot_model_rms_comparison()

    print("Group C: fit-quality diagnostics")
    plot_mass_residuals()
    plot_pulls()
    plot_chi2_rms()

    print("Group D: decay widths & observables")
    plot_width_comparison("Leptonic Width (e+e-)", "D1_leptonic_widths",
                          r"Leptonic $V\to e^+e^-$ widths vs. PDG")
    plot_width_comparison("Two-Photon Width (γγ)", "D2_twophoton_widths",
                          r"Two-photon $P\to\gamma\gamma$ widths vs. PDG")
    plot_radiative_widths()
    plot_rms_radii()
    plot_wavefunction_at_origin()

    print("Group E: physics inputs")
    plot_cornell_potentials(params)
    plot_running_coupling(params)
    plot_cornell_literature(params)

    print(f"\nDone. {len(_saved)} figures written to figures/ (PDF + PNG).")


if __name__ == "__main__":
    main()
