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
    "Deng 2017 (scr. GI)": ("#8030a0", "^"),
    "Radford-Repko (Cornell)": ("#17a2b8", "v"),
    "BGS 2005 (NR)": ("#b22222", "P"),
    "BGS 2005 (GI)": ("#e377c2", "*"),
    "Godfrey-Moats 2015 (GI)": ("#b22222", "*"),
}
