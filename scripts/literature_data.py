"""Theoretical mass predictions from the literature, for model-comparison plots.

All masses are in MeV. Keys are spectroscopic labels ``"n 2S+1 L J"`` (e.g.
``"1 3S1"``, ``"1 1P1"``, ``"1 3D1"``), which match the canonical keys produced by
:func:`make_figures.canonical_key` so that this work's states line up with the
literature row-by-row.

``CHARMONIUM`` and ``BOTTOMONIUM`` are ``{model_label: {state_key: mass_MeV}}``.
Add or drop a model by editing one dict; the comparison plots adapt automatically.

Sources
-------
BGS 2005   : T. Barnes, S. Godfrey, E. S. Swanson, "Higher Charmonia",
    Phys. Rev. D 72, 054026 (2005), arXiv:hep-ph/0505002. Table I, columns
    "NR" (nonrelativistic Cornell-type) and "GI" (relativized Godfrey-Isgur).
    [charmonium; read from the primary PDF]
GM 2015    : S. Godfrey, K. Moats, "Bottomonium Mesons and Strategies for Their
    Observation", Phys. Rev. D 92, 054034 (2015), arXiv:1507.00024. Table I, GI
    column. [bottomonium; read from the primary HTML]
Soni 2018  : N. R. Soni et al., "QQbar (Q in {b,c}) spectroscopy using the
    Cornell potential", Eur. Phys. J. C 78, 592 (2018), arXiv:1707.07144.
    Nonrelativistic Cornell potential (closest methodology to this work).
EFG 2011   : D. Ebert, R. N. Faustov, V. O. Galkin, "Spectroscopy and Regge
    trajectories of heavy quarkonia and Bc mesons", Eur. Phys. J. C 71, 1825
    (2011), arXiv:1111.0454. Relativistic quasipotential quark model.
Deng 2017  : W.-J. Deng, H. Liu, L.-C. Gui, X.-H. Zhong, Phys. Rev. D 95,
    034026 (charmonium) / 074002 (bottomonium) (2017). Relativized GI model with
    a screened linear potential.
RR         : S. F. Radford, W. W. Repko, Phys. Rev. D 75, 074031 (2007)
    (charmonium) / Nucl. Phys. A 865, 69 (2011) (bottomonium). Cornell potential
    with full one-loop QCD radiative corrections.

The Soni/EFG/Deng/RR values are taken from the state-aligned comparison
Tables II-V of Soni 2018 (which collects them against a common state assignment),
converted GeV -> MeV. BGS and GM are read from their own primary papers.

Additional sources for the B_c, D, decay-width, radiative-transition, and
wavefunction-at-origin tables added below
--------------------------------------------------------------------------------
EFG 2003   : D. Ebert, R. N. Faustov, V. O. Galkin, Phys. Rev. D 67, 014027
    (2003), arXiv:hep-ph/0210381. Relativistic quark model (the comparison
    column [30] of Soni 2018 for charmonium/bottomonium transitions).
EFG (D)    : D. Ebert, R. N. Faustov, V. O. Galkin, "Heavy-light meson
    spectroscopy and Regge trajectories in the relativistic quark model",
    Eur. Phys. J. C 66, 197 (2010), arXiv:0910.5612. Source of the D (c u-bar)
    masses. [read from the primary ar5iv HTML, Table 1]
Godfrey 2004 : S. Godfrey, "Spectroscopy of B_c mesons in the relativized quark
    model", Phys. Rev. D 70, 054017 (2004), arXiv:hep-ph/0406228. Relativized
    Godfrey-Isgur model for B_c (column [63] of Soni 2018).
DKR 2014   : N. Devlani, V. Kher, A. K. Rai, Eur. Phys. J. A 50, 154 (2014).
    Cornell-potential B_c spectroscopy (column [46] of Soni 2018).
EQ 2019    : E. J. Eichten, C. Quigg, "Quarkonium wave functions at the origin:
    an update", arXiv:1904.11542, FERMILAB-PUB-19/176-T. Frozen-alpha_s potential;
    source of the |R(0)|^2 (S-wave) and |R'(0)|^2 (P-wave) values.
HPQCD 2010 : E. B. Gregory et al. (HPQCD), Phys. Rev. D 83, 014506 (2011),
    arXiv:0909.4462 ("A prediction of the B_c* mass in full lattice QCD"):
    M(B_c*) = 6330(7)(2)(6) MeV with the PDG B_c(0-) = 6275(1) MeV anchor.

Almost every theory number below (B_c masses, leptonic/two-photon/digamma widths,
E1/M1 transition widths) is read directly from the comprehensive comparison tables
of Soni 2018 (arXiv:1707.07144) -- masses Tables II-VII, two-photon Tables XIV-XV,
dilepton Tables XVIII-XIX, E1 Tables XX-XXII, M1 Tables XXIII-XXV -- which align
the major models against one common state assignment. Where a value is read from a
different primary paper it is noted in the per-dict comment.
"""

# =========================== CHARMONIUM (c cbar) =========================== #
CHARMONIUM = {
    "Soni 2018 (Cornell, NR)": {
        "1 1S0": 2989, "1 3S1": 3094, "2 1S0": 3602, "2 3S1": 3681,
        "3 1S0": 4058, "3 3S1": 4129, "1 3P0": 3428, "1 3P1": 3468,
        "1 1P1": 3470, "1 3P2": 3480, "2 3P0": 3897, "2 3P1": 3938,
        "2 1P1": 3943, "2 3P2": 3955, "1 3D1": 3775, "1 3D2": 3772,
        "1 3D3": 3755, "1 1D2": 3765,
    },
    "EFG 2011 (rel.)": {
        "1 1S0": 2981, "1 3S1": 3096, "2 1S0": 3635, "2 3S1": 3685,
        "3 1S0": 3989, "3 3S1": 4039, "1 3P0": 3413, "1 3P1": 3511,
        "1 1P1": 3525, "1 3P2": 3555, "2 3P0": 3870, "2 3P1": 3906,
        "2 1P1": 3926, "2 3P2": 3949, "1 3D1": 3783, "1 3D2": 3795,
        "1 3D3": 3813, "1 1D2": 3807,
    },
    "Deng 2017 (scr. GI)": {
        "1 1S0": 2984, "1 3S1": 3097, "2 1S0": 3637, "2 3S1": 3679,
        "3 1S0": 4004, "3 3S1": 4030, "1 3P0": 3415, "1 3P1": 3521,
        "1 1P1": 3526, "1 3P2": 3553, "2 3P0": 3848, "2 3P1": 3914,
        "2 1P1": 3916, "2 3P2": 3937, "1 3D1": 3792, "1 3D2": 3807,
        "1 3D3": 3808, "1 1D2": 3805,
    },
    "Radford-Repko (Cornell)": {
        "1 1S0": 2980, "1 3S1": 3097, "2 1S0": 3597, "2 3S1": 3685,
        "3 1S0": 4014, "3 3S1": 4095, "1 3P0": 3416, "1 3P1": 3508,
        "1 1P1": 3527, "1 3P2": 3558, "2 3P0": 3844, "2 3P1": 3940,
        "2 1P1": 3960, "2 3P2": 3994, "1 3D1": 3804, "1 3D2": 3824,
        "1 3D3": 3831, "1 1D2": 3824,
    },
    "BGS 2005 (NR)": {
        "1 1S0": 2982, "1 3S1": 3090, "2 1S0": 3630, "2 3S1": 3672,
        "3 3S1": 4072, "3 1S0": 4043, "1 3P0": 3424, "1 3P1": 3505,
        "1 1P1": 3516, "1 3P2": 3556, "2 3P0": 3852, "2 3P1": 3925,
        "2 1P1": 3934, "2 3P2": 3972, "1 3D1": 3785, "1 3D2": 3800,
        "1 3D3": 3806, "1 1D2": 3799,
    },
    "BGS 2005 (GI)": {
        "1 1S0": 2975, "1 3S1": 3098, "2 1S0": 3623, "2 3S1": 3676,
        "3 3S1": 4100, "3 1S0": 4064, "1 3P0": 3445, "1 3P1": 3510,
        "1 1P1": 3517, "1 3P2": 3550, "2 3P0": 3916, "2 3P1": 3953,
        "2 1P1": 3956, "2 3P2": 3979, "1 3D1": 3819, "1 3D2": 3838,
        "1 3D3": 3849, "1 1D2": 3837,
    },
}

# =========================== BOTTOMONIUM (b bbar) ========================== #
BOTTOMONIUM = {
    "Soni 2018 (Cornell, NR)": {
        "1 1S0": 9428, "1 3S1": 9463, "2 1S0": 9955, "2 3S1": 9979,
        "3 1S0": 10338, "3 3S1": 10359, "1 3P0": 9806, "1 3P1": 9819,
        "1 1P1": 9821, "1 3P2": 9825, "2 3P0": 10205, "2 3P1": 10217,
        "2 1P1": 10220, "2 3P2": 10224, "1 3D1": 10074, "1 3D2": 10075,
        "1 3D3": 10073, "1 1D2": 10074,
    },
    "EFG 2011 (rel.)": {
        "1 1S0": 9398, "1 3S1": 9460, "2 1S0": 9990, "2 3S1": 10023,
        "3 1S0": 10329, "3 3S1": 10355, "1 3P0": 9859, "1 3P1": 9892,
        "1 1P1": 9900, "1 3P2": 9912, "2 3P0": 10233, "2 3P1": 10255,
        "2 1P1": 10260, "2 3P2": 10268, "1 3D1": 10154, "1 3D2": 10161,
        "1 3D3": 10166, "1 1D2": 10163,
    },
    "Deng 2017 (scr. GI)": {
        "1 1S0": 9390, "1 3S1": 9460, "2 1S0": 9990, "2 3S1": 10015,
        "3 1S0": 10326, "3 3S1": 10343, "1 3P0": 9864, "1 3P1": 9903,
        "1 1P1": 9909, "1 3P2": 9921, "2 3P0": 10220, "2 3P1": 10249,
        "2 1P1": 10254, "2 3P2": 10264, "1 3D1": 10146, "1 3D2": 10153,
        "1 3D3": 10157, "1 1D2": 10153,
    },
    "Radford-Repko (Cornell)": {
        "1 1S0": 9393, "1 3S1": 9460, "2 1S0": 9987, "2 3S1": 10023,
        "3 1S0": 10345, "3 3S1": 10364, "1 3P0": 9861, "1 3P1": 9891,
        "1 1P1": 9900, "1 3P2": 9912, "2 3P0": 10230, "2 3P1": 10255,
        "2 1P1": 10262, "2 3P2": 10271, "1 3D1": 10149, "1 3D2": 10157,
        "1 3D3": 10163, "1 1D2": 10158,
    },
    "Godfrey-Moats 2015 (GI)": {
        "1 1S0": 9402, "1 3S1": 9465, "2 1S0": 9976, "2 3S1": 10003,
        "3 3S1": 10354, "1 3P0": 9847, "1 3P1": 9876, "1 1P1": 9882,
        "1 3P2": 9897, "2 3P0": 10226, "2 3P1": 10246, "2 3P2": 10261,
        "1 3D1": 10138, "1 3D2": 10147, "1 3D3": 10155,
    },
}

# ============================ B_c MESON (b c-bar) ========================== #
# Masses in MeV, keyed "n 2S+1 L J" like the quarkonium dicts. Read from
# Soni 2018 Tables VI-VII (column "Present" = Soni Cornell NR), and the aligned
# comparison columns therein: [46]=DKR 2014, [27]=EFG 2011, [63]=Godfrey 2004.
# (Soni gives B_c masses in GeV to 3 d.p.; values below are those x1000.)
BC_MESON = {
    "Soni 2018 (Cornell, NR)": {
        "1 1S0": 6272, "1 3S1": 6321, "2 1S0": 6864, "2 3S1": 6900,
        "3 1S0": 7306, "3 3S1": 7338,
        "1 3P0": 6686, "1 3P1": 6705, "1 1P1": 6706, "1 3P2": 6712,
        "2 3P0": 7146, "2 3P1": 7165, "2 1P1": 7168, "2 3P2": 7173,
        "1 3D1": 6998, "1 3D2": 6997, "1 3D3": 6990, "1 1D2": 6994,
    },
    "EFG 2011 (rel.)": {
        "1 1S0": 6272, "1 3S1": 6333, "2 1S0": 6842, "2 3S1": 6882,
        "3 1S0": 7226, "3 3S1": 7258,
        "1 3P0": 6699, "1 3P1": 6750, "1 1P1": 6743, "1 3P2": 6761,
        "2 3P0": 7094, "2 3P1": 7134, "2 1P1": 7094, "2 3P2": 7157,
        "1 3D1": 7021, "1 3D2": 7025, "1 3D3": 7029, "1 1D2": 7026,
    },
    "Godfrey 2004 (rel. GI)": {
        "1 1S0": 6271, "1 3S1": 6338, "2 1S0": 6855, "2 3S1": 6887,
        "3 1S0": 7250, "3 3S1": 7272,
        "1 3P0": 6706, "1 3P1": 6741, "1 1P1": 6750, "1 3P2": 6768,
        "2 3P0": 7122, "2 3P1": 7145, "2 1P1": 7150, "2 3P2": 7164,
        "1 3D1": 7028, "1 3D2": 7036, "1 3D3": 7045, "1 1D2": 7041,
    },
    "DKR 2014 (Cornell)": {
        "1 1S0": 6278, "1 3S1": 6331, "2 1S0": 6863, "2 3S1": 6873,
        "3 1S0": 7244, "3 3S1": 7249,
        "1 3P0": 6748, "1 3P1": 6767, "1 1P1": 6769, "1 3P2": 6775,
        "2 3P0": 7139, "2 3P1": 7155, "2 1P1": 7156, "2 3P2": 7162,
        "1 3D1": 7030, "1 3D2": 7025, "1 3D3": 7026, "1 1D2": 7035,
    },
}

# Lattice QCD anchor for the B_c ground-state doublet. M(B_c) is the PDG-anchored
# 0- input; M(B_c*) is the HPQCD full-lattice prediction (Gregory et al. 2011).
# (mass_MeV, +/- MeV).  Useful as a horizontal reference on B_c spectrum plots.
BC_LATTICE = {
    "1 1S0": (6275, 1),     # PDG/HPQCD anchor, B_c(1S) pseudoscalar
    "1 3S1": (6330, 7),     # HPQCD prediction, B_c*(1S) vector
}

# =========================== LEPTONIC WIDTHS =============================== #
# Gamma(V -> e+e-) in keV, keyed "n 3S1" (the vector S-wave states only).
# Read from Soni 2018 Tables XVIII (charmonia) / XIX (bottomonia). The "Present"
# column is Soni's Cornell-NR result; comparison columns are identified per the
# Soni reference list:
#   cc Table XVIII: [73]=Shah-Parmar-Vinodkumar 2012, [52]=Patel-Vinodkumar 2009,
#                   [39]=Radford-Repko 2007, [31]=EFG 2003.
#   bb Table XIX:   [73]=Shah et al., [40]=Radford-Repko 2011, [52]=Patel-V.,
#                   [31]=EFG 2003, [123]=Gonzalez et al. 2003.
LEPTONIC_WIDTHS = {
    "cc": {
        "Soni 2018 (Cornell, NR)": {"1 3S1": 2.925, "2 3S1": 1.533,
                                    "3 3S1": 1.091},
        "Shah 2012":               {"1 3S1": 4.95,  "2 3S1": 1.69,
                                    "3 3S1": 0.96},
        "Patel-Vinodkumar 2009":   {"1 3S1": 6.99,  "2 3S1": 3.38,
                                    "3 3S1": 2.31},
        "Radford-Repko (Cornell)": {"1 3S1": 1.89,  "2 3S1": 1.04,
                                    "3 3S1": 0.77},
        "EFG 2003 (rel.)":         {"1 3S1": 5.4,   "2 3S1": 2.4},
    },
    "bb": {
        "Soni 2018 (Cornell, NR)": {"1 3S1": 1.098, "2 3S1": 0.670,
                                    "3 3S1": 0.541},
        "Shah 2012":               {"1 3S1": 1.20,  "2 3S1": 0.52,
                                    "3 3S1": 0.33},
        "Radford-Repko (Cornell)": {"1 3S1": 1.33,  "2 3S1": 0.62,
                                    "3 3S1": 0.48},
        "Patel-Vinodkumar 2009":   {"1 3S1": 1.61,  "2 3S1": 0.87,
                                    "3 3S1": 0.66},
        "EFG 2003 (rel.)":         {"1 3S1": 1.3,   "2 3S1": 0.5},
        "Gonzalez 2003":           {"1 3S1": 0.98,  "2 3S1": 0.41,
                                    "3 3S1": 0.27},
    },
}

# ========================== TWO-PHOTON WIDTHS ============================== #
# Gamma(P -> gamma gamma) in keV for pseudoscalars (n 1S0) and the P-wave
# triplets chi_0 (n 3P0) and chi_2 (n 3P2). chi_1 (3P1) is forbidden by Yang's
# theorem. Read from Soni 2018 Tables XIV (charmonia) / XV (bottomonia).
#   cc Table XIV: [76]=Li-Chao 2009, [32]=EFG 2003, [68]=Lakhina-Swanson 2006,
#                 [121]=Kim-Lee-Wang 2005.
#   bb Table XV:  [77]=Li-Chao 2009, [62]=Godfrey-Isgur 1985, [32]=EFG 2003,
#                 [68]=Lakhina-Swanson 2006.
TWO_PHOTON_WIDTHS = {
    "cc": {
        "Soni 2018 (Cornell, NR)": {"1 1S0": 7.231, "2 1S0": 5.507,
                                    "1 3P0": 8.982, "1 3P2": 1.069,
                                    "2 3P0": 9.111, "2 3P2": 1.084},
        "Li-Chao 2009":            {"1 1S0": 8.5,   "2 1S0": 2.4,
                                    "1 3P0": 2.5,   "1 3P2": 0.31,
                                    "2 3P0": 1.7,   "2 3P2": 0.23},
        "EFG 2003 (rel.)":         {"1 1S0": 5.5,   "2 1S0": 1.8,
                                    "1 3P0": 2.9,   "1 3P2": 0.50,
                                    "2 3P0": 1.9,   "2 3P2": 0.52},
        "Lakhina-Swanson 2006":    {"1 1S0": 7.18,  "2 1S0": 1.71,
                                    "1 3P0": 3.28},
    },
    "bb": {
        "Soni 2018 (Cornell, NR)": {"1 1S0": 0.387, "2 1S0": 0.263,
                                    "1 3P0": 0.0196, "1 3P2": 0.0052,
                                    "2 3P0": 0.0195, "2 3P2": 0.0052},
        "Li-Chao 2009":            {"1 1S0": 0.527, "2 1S0": 0.263,
                                    "1 3P0": 0.050, "1 3P2": 0.0066,
                                    "2 3P0": 0.037, "2 3P2": 0.0067},
        "Godfrey-Isgur 1985":      {"1 1S0": 0.214, "2 1S0": 0.121,
                                    "1 3P0": 0.0208, "1 3P2": 0.0051,
                                    "2 3P0": 0.0227, "2 3P2": 0.0062},
        "EFG 2003 (rel.)":         {"1 1S0": 0.35,  "2 1S0": 0.15,
                                    "1 3P0": 0.038, "1 3P2": 0.008},
    },
}

# ===================== RADIATIVE E1 / M1 TRANSITIONS ====================== #
# Widths in keV (M1 bottomonia/B_c are in eV in Soni; converted to keV here for
# a uniform unit). Keyed by a transition string "ni 2Si+1 Li Ji -> nf ... Jf"
# matching the canonical_key form on each side. Read from Soni 2018:
#   E1: Tables XX (cc), XXI (bb), XXII (B_c).
#   M1: Tables XXIII (cc), XXIV (bb, in eV), XXV (B_c, in eV).
# Comparison columns: [39]=Radford-Repko 2007, [30]=EFG 2003, [76]=Li-Chao 2009,
#   [65]/[66]=Deng et al. 2017, [63]=Godfrey 2004, [46]=DKR 2014.
# Only the transitions this project actually computes (1P->1S E1, and 1S/2S->1S
# M1 for cc and bb) are tabulated; extend as needed.
RADIATIVE_WIDTHS = {
    # ---- charmonium E1: chi_cJ(1P) -> J/psi(1S) + gamma ----
    "1 3P0 -> 1 3S1": {
        "Soni 2018 (Cornell, NR)": 112.0, "Radford-Repko (Cornell)": 142.2,
        "EFG 2003 (rel.)": 161.0, "Li-Chao 2009": 167.0,
        "Deng 2017": 284.0, "PDG": 119.5,
    },
    "1 3P1 -> 1 3S1": {
        "Soni 2018 (Cornell, NR)": 146.3, "Radford-Repko (Cornell)": 287.0,
        "EFG 2003 (rel.)": 333.0, "Li-Chao 2009": 354.0,
        "Deng 2017": 306.0, "PDG": 295.0,
    },
    # ---- charmonium M1: (J/psi, psi(2S)) -> eta_c + gamma ----
    "1 3S1 -> 1 1S0": {
        "Soni 2018 (Cornell, NR)": 2.722, "Radford-Repko (Cornell)": 2.7,
        "EFG 2003 (rel.)": 1.05, "Deng 2017": 2.39, "PDG": 1.58,
    },
    "2 3S1 -> 2 1S0": {
        "Soni 2018 (Cornell, NR)": 1.172, "Radford-Repko (Cornell)": 1.2,
        "EFG 2003 (rel.)": 0.99, "Deng 2017": 0.19, "PDG": 0.21,
    },
    # ---- bottomonium E1: chi_bJ(1P) -> Upsilon(1S) + gamma ----
    "1 3P0(bb) -> 1 3S1": {
        "Soni 2018 (Cornell, NR)": 49.53, "Radford-Repko (Cornell)": 22.1,
        "EFG 2003 (rel.)": 42.7, "Deng 2017": 27.5,
    },
    "1 3P1(bb) -> 1 3S1": {
        "Soni 2018 (Cornell, NR)": 54.93, "Radford-Repko (Cornell)": 27.3,
        "EFG 2003 (rel.)": 37.1, "Deng 2017": 31.9,
    },
    # ---- bottomonium M1: Upsilon(1S,2S) -> eta_b + gamma (Soni eV -> keV) ----
    "1 3S1(bb) -> 1 1S0": {
        "Soni 2018 (Cornell, NR)": 0.0377, "Radford-Repko (Cornell)": 0.0040,
        "EFG 2003 (rel.)": 0.0058, "Deng 2017": 0.01536,
    },
    "2 3S1(bb) -> 2 1S0": {
        "Soni 2018 (Cornell, NR)": 0.00562, "Radford-Repko (Cornell)": 0.00005,
        "EFG 2003 (rel.)": 0.00140, "Deng 2017": 0.00182,
    },
}

# Experimental references for the radiative transitions (keV), used to overlay an
# "Experiment" point. None where no firm measurement exists. PDG 2024 values,
# obtained as B(transition) x Gamma_total of the initial state.
RADIATIVE_WIDTHS_EXP = {
    "1 3P0 -> 1 3S1": 116.0,        # chi_c0 -> J/psi gamma
    "1 3P1 -> 1 3S1": 288.0,        # chi_c1 -> J/psi gamma
    "1 3P2 -> 1 3S1": 406.0,        # chi_c2 -> J/psi gamma (PDG world avg ~406(31) keV)
    "1 3S1 -> 1 1S0": 1.58,         # J/psi -> eta_c gamma
    "2 3S1 -> 2 1S0": 0.05,         # psi(2S) -> eta_c(2S) gamma (small)
    "1 3P0(bb) -> 1 3S1": 27.5,     # chi_b0(1P) -> Upsilon gamma (no PDG width)
    "1 3P1(bb) -> 1 3S1": 31.1,     # chi_b1(1P) -> Upsilon gamma
    "1 3P2(bb) -> 1 3S1": None,     # chi_b2(1P) -> Upsilon gamma (no clean abs. width)
    "1 3S1(bb) -> 1 1S0": None,
    "2 3S1(bb) -> 2 1S0": None,
}

# ===================== WAVEFUNCTION AT THE ORIGIN ========================= #
# |R(0)|^2 in GeV^3 for the S-wave states, from Eichten-Quigg 2019
# (arXiv:1904.11542, frozen-alpha_s potential). These are the canonical
# reference values that drive every annihilation width.
R0_SQUARED = {
    "cc": {"1S": 1.0952, "2S": 0.6966, "3S": 0.5951},   # EQ 2019, GeV^3
    "bb": {"1S": 5.8588, "2S": 2.8974, "3S": 2.2496},
}

# |R'(0)|^2 in GeV^5 for the P-wave states (drives chi -> gamma gamma).
# EQ 2019 tabulate 2P/3P; the 1P values below are from Eichten-Lane-Quigg
# (PRD 73, 014014 (2006)) for completeness.
RP0_SQUARED = {
    "cc": {"1P": 0.075, "2P": 0.1296, "3P": 0.1767},    # GeV^5
    "bb": {"1P": 1.417, "2P": 1.6057, "3P": 1.8240},
}

# =================== Cornell potential parameters ========================== #
# V(r) = -(4/3) alpha_s / r + b r (+ c).  b is the string tension in GeV^2.
# "kind": "cornell"  -> fixed alpha_s, so V(r) can be drawn directly;
#         "running"  -> alpha_s is scale-dependent (or the potential is
#                       screened/relativized), so only the string tension b is
#                       strictly comparable, not a single alpha_s.
CORNELL_PARAMS = {
    "Cornell (Eichten 1980)": {"alpha_s": 0.39, "b": 0.183, "kind": "cornell"},
    "Godfrey-Isgur 1985":     {"alpha_s": None, "b": 0.18,   "kind": "running"},
    "BGS 2005 (NR)":          {"alpha_s": None, "b": 0.1425, "kind": "running"},
    "BGS 2005 (GI)":          {"alpha_s": None, "b": 0.18,   "kind": "running"},
    "Soni 2018 (cc)":         {"alpha_s": None, "b": 0.18,   "kind": "running"},
    "Soni 2018 (bb)":         {"alpha_s": None, "b": 0.25,   "kind": "running"},
}

# Accepted QCD string tension from lattice gauge theory and Regge phenomenology:
# sigma ~ 0.18 GeV^2  (equivalently sqrt(sigma) ~ 0.42-0.44 GeV).
ACCEPTED_STRING_TENSION = (0.18, 0.02)  # (central, +/- spread) in GeV^2

# Shared plotting style per model (so the same model looks identical in both
# the charmonium and bottomonium figures). (color, marker)
MODEL_STYLE = {
    "This work": ("#1f4e8c", "o"),
    "Soni 2018 (Cornell, NR)": ("#d4880b", "s"),
    "EFG 2011 (rel.)": ("#2e8b57", "D"),
    "EFG 2003 (rel.)": ("#2e8b57", "D"),
    "EFG 2010 (rel.)": ("#2e8b57", "D"),
    "Deng 2017 (scr. GI)": ("#8030a0", "^"),
    "Deng 2017": ("#8030a0", "^"),
    "Radford-Repko (Cornell)": ("#17a2b8", "v"),
    "BGS 2005 (NR)": ("#b22222", "P"),
    "BGS 2005 (GI)": ("#e377c2", "*"),
    "Godfrey-Moats 2015 (GI)": ("#b22222", "*"),
    "Godfrey 2004 (rel. GI)": ("#e377c2", "*"),
    "Godfrey-Isgur 1985": ("#e377c2", "X"),
    "DKR 2014 (Cornell)": ("#d62728", "h"),
    "Li-Chao 2009": ("#9467bd", "<"),
    "Lakhina-Swanson 2006": ("#8c564b", ">"),
    "Shah 2012": ("#bcbd22", "p"),
    "Patel-Vinodkumar 2009": ("#7f7f7f", "d"),
    "Gonzalez 2003": ("#ff7f0e", "1"),
    "PDG": ("#b22222", "*"),
    "PDG (measured)": ("#b22222", "*"),
}

# ===================== MANUSCRIPT CITATION NUMBERS ========================= #
# Reference number [N] of each figure legend entry in the compiled paper
# (tex/apssamp.bib). This lets every literature marker in the figures be read
# straight against the paper's reference list. The numbers are assigned by
# bibtex in order of first \cite, so if the manuscript's citation order
# changes, re-read them from tex/apssamp.aux:
#     grep -oP '\\bibcite\{[^}]+\}\{\{[0-9]+\}' tex/apssamp.aux
CITE = {
    # mass-spectrum comparison models
    "Soni 2018 (Cornell, NR)": 9,
    "Soni 2018 (cc)": 9,
    "Soni 2018 (bb)": 9,
    "EFG 2011 (rel.)": 23,
    "EFG 2003 (rel.)": 31,
    "EFG 2010 (rel.)": 27,
    "Deng 2017 (scr. GI)": 24,
    "Deng 2017": 24,
    "Radford-Repko (Cornell)": 25,
    "BGS 2005 (NR)": 8,
    "BGS 2005 (GI)": 8,
    "BGS 2005 (3P0)": 8,
    "Godfrey-Moats 2015 (GI)": 26,
    "Godfrey 2004 (rel. GI)": 28,
    "Godfrey-Isgur 1985": 7,
    "DKR 2014 (Cornell)": 29,
    "Cornell (Eichten 1980)": 4,
    # decay-width comparison models
    "Li-Chao 2009": 35,
    "Lakhina-Swanson 2006": 36,
    "Shah 2012": 32,
    "Patel-Vinodkumar 2009": 33,
    "Gonzalez 2003": 34,
    # measured data / lattice / reference values
    "PDG": 15,
    "PDG 2026": 15,
    "PDG (measured)": 15,
    "PDG (measured total)": 15,
    "Experiment": 15,
    "Lattice (HPQCD)": 22,
    "Eichten-Quigg 2019": 30,
}


def cite_label(label):
    """Append the manuscript reference number to a figure legend label, e.g.
    ``'Soni 2018 (Cornell, NR)'`` -> ``'Soni 2018 (Cornell, NR) [9]'``. Labels
    with no numbered reference (``'This work'``, the modified-GI showcase
    point, theory-band annotations, ...) are returned unchanged."""
    n = CITE.get(label)
    return f"{label} [{n}]" if n is not None else label
