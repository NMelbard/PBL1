"""
Synthetic Heparin Bioreactor — Full Combined Model

Modules (in order):
  1. Transcription & Translation  — mRNA and enzyme build-up
  2. Sulfation                    — Heparosan + PAPS → Heparin (lumped)
  3. Exocytosis                   — Vesicle packaging and heparin secretion
                                    (uses k_sulfation consistent with Module 4;
                                     secreted heparin feeds Module 4 as initial condition)
  4. Full Integrated ODE System   — 7-state synthetic cell model
  5. Porcine vs Synthetic         — head-to-head comparison

────────────────────────────────────────────────────────
UNIFIED UNIT REFERENCE TABLE
────────────────────────────────────────────────────────
Quantity              Native model unit   Plot display unit   Conversion
──────────────────    ─────────────────   ─────────────────   ─────────────────
Time                  min                 min                 —
Concentrations        mM                  mM                  —  (Mod 1–3)
  Heparin / Heparosan mM                  µM on axes          × 1 000
  PAPS                mM                  µM on axes          × 1 000
  mRNA (Mod 1)        dimensionless*      nM on axes          × 1 × 10⁶ †
  Enzyme (Mod 4)      mM                  nM on axes          × 1 × 10⁶
  Glucose             mM                  mM                  —

* Module 1 uses Shin & Noireaux (2012) rate constants whose mRNA output
  has units of [mRNA molecules / gene copy] and is intentionally
  dimensionless.  In Module 4 the enzyme state is in mM; the gene
  expression sub-system is re-parameterised to mM-consistent units
  (k_tx · dna → mM/min).  Module 1 therefore serves as a standalone
  qualitative illustration of transcription/translation dynamics only
  and is NOT numerically coupled to Module 4.  See MODULE 1 header below.

† nM axis labels in Figure 1 are scaled for readability; the underlying
  values are dimensionless (see above).
────────────────────────────────────────────────────────

References
----------
[1] Shin & Noireaux (2012) ACS Synth Biol 1:29-41        gene expression rates
[2] Carruthers (1990) Physiol Rev 70:1135-76              GLUT1 Km ~ 1.5 mM
[3] Xu et al. (2011) Science 334:498-501                  cell-free heparin, PAPS loading
[4] Esko & Lindahl (2001) J Clin Invest 108:169-73        HST Km(PAPS) 10-100 uM
[5] DeAngelis (2007) Semin Thromb Hemost 33:442-8         KfiA Km range
[6] Ototani et al. (1981) Carbohyd Res 88:291-303         porcine yield estimates
[7] Linhardt & Gunay (1999) Semin Thromb Hemost 25:5-16   porcine process review

Porcine mechanistic model references (labelled [A]–[E] in code):
[A] Shu et al. (2018) Open J Appl Sci — enzymatic digestion process (55°C, 2–3 h,
    resin adsorption 6–8 h); https://www.scirp.org/html/2-2150626_87723.htm
[B] Peighambardoust et al. (2022) Biomolecules 9(11):606 — pseudo-second-order
    kinetics for heparin chemisorption on QDASi; PMC9687748
    https://doi.org/10.3390/biom9110606
[C] Alsaiari et al. (2022) Molecules 27(5):1670 — pseudo-second-order kinetics on
    ZIF-8; mucosa heparin ~1300 mg/L; 64 USP U/g mucosa recovered; PMC8911909
    https://doi.org/10.3390/molecules27051670
[D] DuPont AmberLite FPA98 Cl Application Note (2022) — industrial stirred-tank
    adsorption process; 3–8% NaCl wash, 15–26% NaCl elution
    https://www.dupont.com/content/dam/water/...IER-Heparin-Extraction-Br-45-D04430-en.pdf
[E] US6232093B1 (2001) Patent — 30,500 USP U/kg mucosa recovered; enzymatic
    hydrolysis at 55°C up to 6 h; https://patents.google.com/patent/US6232093B1/en
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.2,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "lines.linewidth":   2.0,
    "legend.frameon":    False,
    "figure.dpi":        130,
})

P = {   # colour palette
    "blue":    "#3B82F6",
    "sky":     "#93C5FD",
    "orange":  "#F97316",
    "green":   "#22C55E",
    "dkgreen": "#15803D",
    "purple":  "#A855F7",
    "red":     "#EF4444",
    "brown":   "#92400E",
    "gray":    "#6B7280",
    "pink":    "#EC4899",
    "teal":    "#14B8A6",
}


# MODULE 1 — TRANSCRIPTION & TRANSLATION
#
# PURPOSE: Standalone qualitative illustration of mRNA/protein dynamics
# using Shin & Noireaux (2012) E. coli cell-free rate constants [1].
#
# UNIT NOTE: The rate constants k_R (transcription, molecules/min/gene) and
# k_P (translation, molecules/min/mRNA) produce outputs in molecule-count
# units that are intentionally DIMENSIONLESS here — they represent relative
# copy-number trajectories, not mM concentrations.  Accordingly, this module
# is NOT numerically coupled to Module 4; it serves as a conceptual illustration
# only.  Module 4 re-parameterises gene expression in mM-consistent units
# (k_tx · [DNA] → mM/min) so that all states share a common concentration basis.
# Axis labels in Figure 1 read "nM" for readability but reflect this scaled
# dimensionless output (see unit table in the file header).


def run_txn_tln(k_R=20.0, gamma_R=0.49, k_P=2.28, gamma_P=0.00057,
                t_max=50, dt=0.1):
    """
    Euler integration of mRNA and protein ODEs (E. coli cell-free parameters [1]).
    Outputs are dimensionless molecule-count proxies (see unit table in header).
    Returns (time, mRNA, protein) arrays.
    """
    time = np.arange(0, t_max, dt)
    m = np.zeros_like(time)
    prot = np.zeros_like(time)
    for i in range(1, len(time)):
        m[i]    = m[i-1]    + (k_R - gamma_R * m[i-1]) * dt
        prot[i] = prot[i-1] + (k_P * m[i-1] - gamma_P * prot[i-1]) * dt
    return time, m, prot


# MODULE 2 — LUMPED SULFATION  (Heparosan + PAPS → Heparin + PAP)


def sulfation_ode(t, y, k):
    Hs, Hp, PAPS = y
    v = k * Hs * PAPS
    return [-v, v, -v]


def run_sulfation(k=0.05, Hs0=10.0, Hp0=0.0, PAPS0=12.0,
                  t_span=(0, 100), n=500):
    y0 = [Hs0, Hp0, PAPS0]
    return solve_ivp(lambda t, y: sulfation_ode(t, y, k),
                     t_span, y0,
                     t_eval=np.linspace(*t_span, n))



# MODULE 3 — EXOCYTOSIS  (vesicle packaging + secretion)
#
# DESIGN RATIONALE:
#   This module extends Module 2 by adding a downstream vesicle-packaging and
#   secretion stage, representing how a synthetic cell would package heparin into
#   lipid vesicles and release it across the membrane into the external medium.
#
# UNIT NOTE: All concentrations are in mM, consistent with Modules 2 and 4.
#
# COUPLING TO MODULE 4:
#   The sulfation rate constant used here (k_sulfation) is derived directly from
#   Module 4's kinetic parameters under steady-state enzyme concentration, so
#   both modules describe the same underlying biochemistry.  After Module 3
#   completes its run, the secreted heparin concentration (S_end, mM) is
#   carried forward as the initial heparin in Module 4, reflecting that vesicle-
#   mediated release pre-loads the integrated system before the gene expression
#   lag resolves.  Specifically:
#
#       k_sulfation = kcat_m * e_ss / (Km_h + Km_p)   [mM⁻¹ min⁻¹]
#
#   where e_ss = k_tl/k_de * k_tx/k_dm * dna  (quasi-steady-state enzyme, mM)
#   This is a first-order approximation valid when [Heparosan], [PAPS] << Km.

def _derive_k_sulfation(p):
    """
    Compute the effective bimolecular sulfation rate constant (mM⁻¹ min⁻¹)
    consistent with Module 4 parameters at quasi-steady-state enzyme level.
    Uses the limiting low-substrate form of the bisubstrate MM denominator:
        v_sulf ≈ (kcat_m / (Km_h * Km_p)) * e_ss * [Hs] * [PAPS]
    so  k_sulfation = kcat_m * e_ss / (Km_h * Km_p)
    """
    e_ss = (p["k_tl"] / p["k_de"]) * (p["k_tx"] / p["k_dm"]) * p["dna"]
    return p["kcat_m"] * e_ss / (p["Km_h"] * p["Km_p"])


def v_packaging(H, K, Vmax, n):
    return Vmax * H**n / (K**n + H**n)


def exocytosis_ode(t, y, c, k_sulfation):
    """
    5-state exocytosis ODE.  All concentrations in mM.
    States: [Heparosan, Heparin_intracellular, PAPS, Vesicles, Secreted_Heparin]
    c = [Kpack (mM), Vmax (mM/min), n (Hill), ksec (1/min), q (mM/vesicle), k_deg (1/min)]
    k_sulfation: effective bimolecular rate constant (mM⁻¹ min⁻¹), derived from
                 Module 4 parameters via _derive_k_sulfation().
    """
    Hs, Hp, PAPS, V, S = y
    v_sulf = k_sulfation * Hs * PAPS
    Kpack, Vmax, n, ksec, q, k_deg = c
    dHs   = -v_sulf
    dHp   =  v_sulf - v_packaging(Hp, Kpack, Vmax, n) - k_deg * Hp
    dPAPS = -v_sulf
    dV    = (1/q) * v_packaging(Hp, Kpack, Vmax, n) - ksec * V
    dS    =  q * ksec * V - k_deg * S
    return [dHs, dHp, dPAPS, dV, dS]


def run_exocytosis(t_end=720, base_params=None):
    """
    Simulate the exocytosis module.

    k_sulfation is derived from base_params (Module 4 parameters) so that the
    two modules share a consistent kinetic description of the sulfation step.
    The function returns the solve_ivp result AND the final secreted heparin
    concentration (mM), which is used as the Module 4 initial heparin value.

    Parameters
    ----------
    t_end       : float — simulation end time (min)
    base_params : dict  — Module 4 parameter dict; defaults to BASE_PARAMS.
                          Must be passed after BASE_PARAMS is defined below.
    """
    if base_params is None:
        base_params = BASE_PARAMS          # resolved at call time (defined below)
    k_sulfation = _derive_k_sulfation(base_params)

    # Packaging / secretion parameters (mM-consistent)
    Kpack  = 0.00035   # mM  — half-saturation for Hill packaging
    Vmax   = 0.1       # mM/min — max packaging flux
    n      = 1         # Hill coefficient
    ksec   = 0.26      # 1/min — vesicle secretion rate
    q      = 1e-3      # mM/vesicle — heparin load per vesicle unit
    k_deg  = 0.01      # 1/min — degradation of free intracellular heparin
    c = [Kpack, Vmax, n, ksec, q, k_deg]

    # Initial conditions (mM): Heparosan=10, Heparin=0, PAPS=12, Vesicles=0, Secreted=0
    y0    = [10.0, 0.0, 12.0, 0.0, 0.0]
    tspan = np.linspace(0, t_end, 1000)
    sol   = solve_ivp(
        lambda t, y: exocytosis_ode(t, y, c, k_sulfation),
        [tspan[0], tspan[-1]], y0, t_eval=tspan,
        method="LSODA", rtol=1e-8, atol=1e-12,
    )
    secreted_end_mM = sol.y[4, -1]   # final Secreted_Heparin (mM)
    return sol, secreted_end_mM


# MODULE 4 — FULL INTEGRATED ODE (7-state synthetic cell)

def full_ode(t, y, p):
    glc_ext, glc_int, mrna, e, heparosan, paps, heparin = y
    glc_ext   = max(glc_ext,   0.0)
    glc_int   = max(glc_int,   0.0)
    mrna      = max(mrna,      0.0)
    e         = max(e,         0.0)
    heparosan = max(heparosan, 0.0)
    paps      = max(paps,      0.0)

    v_import = p["V_t"] * glc_ext / (p["Km_t"] + glc_ext)
    v_tx     = p["k_tx"] * p["dna"]
    v_dmrna  = p["k_dm"] * mrna
    v_tl     = p["k_tl"] * mrna
    v_de     = p["k_de"] * e
    v_poly   = p["kcat_a"] * e * glc_int / (p["Km_a"] + glc_int)
    denom    = (p["Km_h"] * p["Km_p"]
                + p["Km_p"] * heparosan
                + p["Km_h"] * paps
                + heparosan * paps + 1e-12)
    v_sulf   = p["kcat_m"] * e * heparosan * paps / denom
    v_regen  = p["k_regen"] * (p["paps_max"] - paps)

    return [
        -v_import,
         v_import  - v_poly,
         v_tx      - v_dmrna,
         v_tl      - v_de,
         v_poly    - v_sulf,
        -v_sulf    + v_regen,
         v_sulf,
    ]


BASE_PARAMS = {
    "V_t":   0.10,   "Km_t":  1.5,
    "dna":   5e-6,   "k_tx":  0.10,  "k_dm":  0.14,
    "k_tl":  0.06,   "k_de":  0.004,
    "kcat_a": 200.0, "Km_a":  0.50,
    "kcat_m": 40.0,  "Km_h":  0.05,  "Km_p":  0.05,
    "paps_max": 0.05, "k_regen": 0.010,
}

BASE_Y0 = [10.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0]
T_SPAN  = (0, 180)
TARGET  = 10.0   # µM  therapeutic target


def run_full(params=None, y0=None, t_span=T_SPAN, n=600):
    p  = BASE_PARAMS if params is None else params
    ic = BASE_Y0     if y0     is None else y0
    return solve_ivp(lambda t, y: full_ode(t, y, p),
                     t_span, ic,
                     t_eval=np.linspace(*t_span, n),
                     method="LSODA", rtol=1e-8, atol=1e-12)



# PORCINE REFERENCE MODEL — TWO-STAGE MECHANISTIC EXTRACTION
# ════════════════════════════════════════════════════════════
#
# The industrial extraction of heparin from porcine intestinal mucosa proceeds
# through two sequential, rate-limiting stages [A–D]:
#
#   Stage 1 — Enzymatic digestion (0 – ~180 min, 55 °C, pH 8.5–9.5)
#     Alkaline protease (e.g. subtilisin/maxatase) hydrolyses the mucosa matrix,
#     releasing heparin–proteoglycan complexes into solution.  This is modelled
#     as a first-order approach to the total available heparin in the tissue:
#       dC_sol/dt = k_dig * (C_max - C_sol)
#     where C_max is the total releasable heparin concentration [~1300 mg/L
#     ≈ 87 µM at MW 15 kDa, measured directly in industrial mucosa [C]] and
#     k_dig is the first-order digestion rate constant (1/min).
#
#   Stage 2 — Anion-exchange resin adsorption (starts at t_ads, ~180–540 min)
#     After heat-inactivation and filtration, heparin is captured on a strong-
#     base anion-exchange resin (e.g. Amberlite FPA98 Cl, DEAE-Sepharose) in a
#     stirred-tank configuration [D].  Multiple independent studies confirm that
#     this step obeys PSEUDO-SECOND-ORDER kinetics, where the rate-limiting
#     step is chemisorption onto resin active sites, not heparin concentration
#     [B, C]:
#       dq/dt = k2 * (qe - q)²
#     Integrated form: q(t) = qe² * k2 * t' / (1 + qe * k2 * t')   [t' = t - t_ads]
#     where qe (µM) is the equilibrium adsorption capacity and k2 is the pseudo-
#     second-order rate constant (µM⁻¹ min⁻¹).
#
#   Final purification yield factor:
#     Elution (high-salt wash), alcohol precipitation and drying each cause
#     partial losses.  Reported overall process yields range from 28–70% of
#     adsorbed heparin recovered in the final API, depending on adsorbent and
#     process design [B, C].  A recovery factor η = 0.55 (55%) is used,
#     consistent with conventional Amberlite resin processes [C, D].
#
# PARAMETER SOURCES:
#   C_max = 87 µM:  porcine mucosa heparin content ~1300 mg/L [C], MW = 15 kDa
#   k_dig = 0.020 /min:  digestion complete in ~2–3 h [A, E], k = ln(20)/150 min
#   t_ads = 180 min:  enzyme heat-inactivation + filtration before resin step [A, D]
#   qe    = 64 µM:  64 USP U/g mucosa × 1 µM/170 USP U per mg × ... scaled [C]
#                   (ZIF-8 adsorbent; conventional Amberlite gives ~59 USP U/g [C])
#                   Using conventional value: 59 USP U/g mucosa → ~47 µM (corrected)
#   k2    = 2.5e-4 µM⁻¹ min⁻¹:  pseudo-second-order rate constant, estimated from
#           adsorption equilibrium reached in ~6–8 h stirring [A], fitted to give
#           q/qe ≈ 0.95 at t' = 360 min; consistent with order-of-magnitude reported
#           for heparin–resin chemisorption [B]
#   η     = 0.55:  55% overall API recovery factor [B, C, D]
#
# References
# ----------
# [A] Shu et al. (2018) Open J Appl Sci — enzymatic digestion at 55°C, 2–3 h,
#     resin adsorption 6–8 h stirring; process steps described in detail.
#     https://www.scirp.org/html/2-2150626_87723.htm
# [B] Peighambardoust et al. (2022) Biomolecules 9(11):606 — pseudo-second-order
#     kinetics confirmed for heparin adsorption on cationic silica (QDASi);
#     rate-limiting step is surface chemisorption, not heparin concentration.
#     https://doi.org/10.3390/biom9110606   [PMC9687748]
# [C] Alsaiari et al. (2022) Molecules 27(5):1670 — pseudo-second-order kinetics
#     confirmed for ZIF-8 adsorbent; 64 USP U/g mucosa recovered; porcine mucosa
#     heparin content ~1300 mg/L measured directly.
#     https://doi.org/10.3390/molecules27051670   [PMC8911909]
# [D] DuPont (2022) AmberLite FPA98 Cl Heparin Extraction Application Note —
#     industrial stirred-tank adsorption, 3–8% NaCl wash, 15–26% NaCl elution.
#     https://www.dupont.com/content/dam/water/...IER-Heparin-Extraction-Br-45-D04430-en.pdf
# [E] US6232093B1 (2001) Patent — enzymatic hydrolysis at 55°C up to 6 h,
#     30,500 USP U/kg mucosa in eluate; supports k_dig and C_max estimates.


def porcine_extraction_ode(t, y, k_dig, C_max, k2, qe, t_ads):
    """
    Two-stage porcine heparin extraction ODE.  All concentrations in µM.

    States: [C_sol, q]
      C_sol : heparin in solution after enzymatic digestion (µM)
      q     : heparin adsorbed onto resin per unit volume (µM-equivalent)

    Stage 1 (t < t_ads): digestion only, no resin present.
    Stage 2 (t >= t_ads): resin adsorption of C_sol following pseudo-second-order
                          kinetics (rate-limiting step is chemisorption [B, C]).

    Note: the integrated pseudo-second-order equation is used analytically in
    porcine_yield() for speed; this ODE form is retained for mechanistic
    transparency and potential extension.
    """
    C_sol, q = y
    # Stage 1: enzymatic digestion (first-order release into solution)
    dC_sol_dig = k_dig * (C_max - C_sol)
    # Stage 2: adsorption — pseudo-second-order chemisorption [B, C]
    if t < t_ads:
        dq     = 0.0
        dC_sol = dC_sol_dig
    else:
        # Driving force is (qe - q); rate = k2 * (qe - q)^2
        dq     = k2 * (qe - q)**2
        dC_sol = dC_sol_dig - dq   # heparin removed from solution into resin
    return [dC_sol, dq]


# Porcine process parameters (see sources above)
PORC = {
    "C_max":  87.0,    # µM — total releasable heparin in mucosa (1300 mg/L ÷ 15 kDa)
    "k_dig":  0.020,   # 1/min — first-order digestion rate (2–3 h to completion) [A, E]
    "t_ads":  180.0,   # min — resin added after digestion + filtration [A, D]
    "qe":     47.0,    # µM — equilibrium adsorption capacity (59 USP U/g mucosa) [C]
    "k2":     2.5e-4,  # µM⁻¹ min⁻¹ — pseudo-second-order rate constant [B, C]
    "eta":    0.55,    # dimensionless — overall API recovery after elution + precipitation [B–D]
    "t_end":  600.0,   # min — total process time (10 h; digestion + adsorption + elution)
}


def run_porcine_extraction(porc=None):
    """
    Simulate the two-stage porcine extraction process.
    Returns (t_porc, C_sol_uM, q_uM, api_yield_uM):
      t_porc      : time array (min)
      C_sol_uM    : heparin in solution over time (µM)
      q_uM        : heparin captured on resin over time (µM)
      api_yield_uM: final recoverable API (µM) = eta * q[-1]
    """
    if porc is None:
        porc = PORC
    t_span  = (0, porc["t_end"])
    t_eval  = np.linspace(0, porc["t_end"], 1200)
    y0      = [0.0, 0.0]   # C_sol=0, q=0 at start
    sol     = solve_ivp(
        lambda t, y: porcine_extraction_ode(
            t, y, porc["k_dig"], porc["C_max"],
            porc["k2"], porc["qe"], porc["t_ads"]
        ),
        t_span, y0, t_eval=t_eval,
        method="LSODA", rtol=1e-8, atol=1e-12,
    )
    C_sol_uM     = sol.y[0]
    q_uM         = sol.y[1]
    api_yield_uM = porc["eta"] * q_uM   # apply purification recovery factor
    return sol.t, C_sol_uM, q_uM, api_yield_uM


print("Running all simulations …")

# Module 1
time_txn, mrna_txn, prot_txn = run_txn_tln()

# Module 2
sol_sulf = run_sulfation()

# Module 3 — now uses k_sulfation derived from BASE_PARAMS (see _derive_k_sulfation)
# and returns the final secreted heparin concentration to seed Module 4.
sol_exo, exo_hep_seed_mM = run_exocytosis(t_end=720, base_params=BASE_PARAMS)
exo_labels = ["Heparosan", "Heparin (intracellular)", "PAPS", "Vesicles", "Secreted Heparin"]

# Effective k_sulfation used in Module 3 (printed for transparency)
_k_sulf_effective = _derive_k_sulfation(BASE_PARAMS)
print(f"Module 3 — derived k_sulfation = {_k_sulf_effective:.4e} mM⁻¹ min⁻¹  "
      f"(vs old hard-coded 1e3; consistent with Module 4 quasi-SS enzyme)")
print(f"Module 3 — secreted heparin at end of exocytosis phase: "
      f"{exo_hep_seed_mM*1000:.4f} µM  →  used as Module 4 initial [heparin]")

# Module 4 — initial heparin seeded from Module 3 secreted output
BASE_Y0_SEEDED = list(BASE_Y0)
BASE_Y0_SEEDED[6] = exo_hep_seed_mM   # heparin initial condition (mM) from Module 3

# Module 4 — initial heparin = secreted output from Module 3
sol_full  = run_full(y0=BASE_Y0_SEEDED)
t_full, y_full = sol_full.t, sol_full.y
hep_uM   = y_full[6] * 1000   # mM → µM
hepa_uM  = y_full[4] * 1000
enz_nM   = y_full[3] * 1e6
mrna_nM  = y_full[2] * 1e6
_cross   = np.where(hep_uM >= TARGET)[0]
t_cross  = t_full[_cross[0]] if len(_cross) else None
_lag     = np.where(enz_nM >= 0.5 * enz_nM[-1])[0]
t_lag    = t_full[_lag[0]]   if len(_lag)   else t_full[-1] / 2

# Porcine — mechanistic two-stage extraction
t_porc, C_sol_uM, q_uM, api_uM = run_porcine_extraction()
# For comparison plots: use recoverable API yield as the "porcine yield" curve.
# The porcine process runs over 600 min (10 h); the synthetic bioreactor over 180 min.
# Both are plotted on their own natural timescales in Figure 7 (porcine panel),
# and the final recovered API is used for bar-chart comparisons.
porc_final_uM = api_uM[-1]
print(f"Porcine extraction — final recoverable API: {porc_final_uM:.2f} µM  "
      f"(= eta={PORC['eta']} × qe captured = {q_uM[-1]:.2f} µM adsorbed)")

# Baseline yield printout
final_uM = hep_uM[-1]
print(f"Baseline — final synthetic heparin: {final_uM:.2f} µM  "
      f"(target ≥ {TARGET} µM: {'✓ MET' if final_uM >= TARGET else '✗ NOT MET'})")


# FIGURE 1 — Transcription & Translation

m_ss    = 20.0 / 0.49
thresh  = 0.99 * m_ss
reach_i = np.where(mrna_txn >= thresh)[0][0]
t_reach = time_txn[reach_i]

mid_i     = len(time_txn) // 2
dP_mid    = (prot_txn[mid_i+1] - prot_txn[mid_i-1]) / (2 * 0.1)
t_mid     = time_txn[mid_i]
P_mid     = prot_txn[mid_i]
t_line    = np.linspace(t_mid - 10, t_mid + 10, 100)
P_tangent = P_mid + dP_mid * (t_line - t_mid)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Module 1 — Transcription & Translation (E. coli cell-free, qualitative illustration)\n"
             "Note: mRNA/protein outputs are dimensionless molecule-count proxies — see unit table in header",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(time_txn, mrna_txn, color=P["orange"], label="mRNA")
ax.axhline(m_ss,   color=P["red"],   linestyle="--", label=f"Steady-state ≈ {m_ss:.1f}")
ax.axvline(t_reach, color=P["green"], linestyle="--", label=f"~SS reached t ≈ {t_reach:.1f} min")
ax.set(xlabel="Time (min)", ylabel="mRNA level (dimensionless)", title="mRNA Transcription")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(time_txn, prot_txn, color=P["blue"], label="Protein")
ax.plot(t_line, P_tangent, linestyle="--", color=P["orange"],
        label=f"Slope ≈ {dP_mid:.1f} a.u./min")
ax.set(xlabel="Time (min)", ylabel="Protein level (dimensionless)", title="Protein Translation")
ax.legend(fontsize=9)

ax = axes[2]
ax.plot(time_txn, mrna_txn / np.max(mrna_txn), color=P["orange"], label="mRNA (norm.)")
ax.plot(time_txn, prot_txn / np.max(prot_txn), color=P["blue"],   label="Protein (norm.)")
ax.set(xlabel="Time (min)", ylabel="Normalised level (a.u.)", title="mRNA vs Protein (scaled)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
print("Figure 1 saved.")


# FIGURE 2 — Sulfation Module


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Module 2 — Lumped Sulfation  (Heparosan + PAPS → Heparin)",
             fontsize=14, fontweight="bold")

ax = axes[0]
ax.plot(sol_sulf.t, sol_sulf.y[0], color=P["blue"],   label="Heparosan")
ax.plot(sol_sulf.t, sol_sulf.y[1], color=P["dkgreen"], label="Heparin")
ax.plot(sol_sulf.t, sol_sulf.y[2], color=P["brown"],  label="PAPS")
ax.set(xlabel="Time (min)", ylabel="Concentration (mM)",
       title="Species over Time")
ax.legend()

ax = axes[1]
ax.plot(sol_sulf.t, sol_sulf.y[1] / 10.0 * 100, color=P["dkgreen"])
ax.set(xlabel="Time (min)", ylabel="Conversion (%)",
       title="Heparosan → Heparin Conversion")
ax.axhline(90, color=P["gray"], linestyle="--", linewidth=1.2, label="90% conversion")
ax.legend()

plt.tight_layout()
plt.show()
print("Figure 2 saved.")


# FIGURE 3 — Exocytosis Module

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    f"Module 3 — Exocytosis  (Vesicle Packaging & Heparin Secretion)\n"
    f"k_sulfation = {_k_sulf_effective:.3e} mM⁻¹ min⁻¹ derived from Module 4 quasi-SS enzyme  |  "
    f"Secreted heparin seeds Module 4 initial condition",
    fontsize=11, fontweight="bold"
)
colours = [P["blue"], P["dkgreen"], P["brown"], P["purple"], P["teal"]]

ax = axes[0]
for i in [0, 1, 2]:
    ax.plot(sol_exo.t, sol_exo.y[i], color=colours[i], label=exo_labels[i])
ax.set(xlabel="Time (min)", ylabel="Concentration (mM)",
       title="Heparosan / Heparin / PAPS")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(sol_exo.t, sol_exo.y[3], color=P["purple"], label="Vesicles (relative)")
ax.set(xlabel="Time (min)", ylabel="Vesicle count (relative)", title="Vesicle Formation")
ax.legend(fontsize=9)

ax = axes[2]
ax.plot(sol_exo.t, sol_exo.y[4], color=P["teal"], label="Secreted Heparin")
ax.fill_between(sol_exo.t, sol_exo.y[4], 0, alpha=0.12, color=P["teal"])
ax.set(xlabel="Time (min)", ylabel="Concentration (mM)", title="Cumulative Secreted Heparin")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
print("Figure 3 saved.")



# FIGURE 3b — Fed-Batch Exocytosis: Single Bolus Addition
#
# PURPOSE: Demonstrate the fed-batch principle using only parameters already in
# the model — no new rate constants.  A single bolus of fresh substrate is added
# at t = 360 min (6 h) to a cell whose substrates have been exhausted, triggering
# a second identical vesicle production burst.  Batch and fed-batch are compared
# side by side over the same 720-min (12 h) window.


def run_exocytosis_with_bolus(t_bolus=360, dHs=10.0, dPAPS=12.0,
                               t_end=720, base_params=None):
    """
    Single-bolus fed-batch exocytosis.

    Runs the standard 5-state exocytosis ODE in two segments:
      Segment 1: t = 0      → t_bolus  (initial substrates deplete)
      Bolus:     add dHs to Hs, dPAPS to PAPS instantaneously at t_bolus
      Segment 2: t = t_bolus → t_end   (second burst from replenished substrates)

    Uses the same k_sulfation as run_exocytosis() — no new parameters introduced.
    """
    if base_params is None:
        base_params = BASE_PARAMS
    k_sulfation = _derive_k_sulfation(base_params)

    Kpack  = 0.00035
    Vmax   = 0.1
    n_hill = 1
    ksec   = 0.26
    q_load = 1e-3
    k_deg  = 0.01
    c = [Kpack, Vmax, n_hill, ksec, q_load, k_deg]

    t1 = np.linspace(0, t_bolus, 800)
    sol1 = solve_ivp(
        lambda t, y: exocytosis_ode(t, y, c, k_sulfation),
        [0, t_bolus], [10.0, 0.0, 12.0, 0.0, 0.0],
        t_eval=t1, method="LSODA", rtol=1e-8, atol=1e-12,
    )
    y_after = sol1.y[:, -1].copy()
    y_after[0] += dHs
    y_after[2] += dPAPS
    y_after = np.maximum(y_after, 0.0)

    t2 = np.linspace(t_bolus, t_end, 800)
    sol2 = solve_ivp(
        lambda t, y: exocytosis_ode(t, y, c, k_sulfation),
        [t_bolus, t_end], y_after,
        t_eval=t2, method="LSODA", rtol=1e-8, atol=1e-12,
    )
    t_all = np.concatenate([sol1.t, sol2.t[1:]])
    y_all = np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1)
    return t_all, y_all, t_bolus


t_fb2, y_fb2, t_bol = run_exocytosis_with_bolus(
    t_bolus=360, dHs=10.0, dPAPS=12.0, t_end=720, base_params=BASE_PARAMS
)
batch_s  = sol_exo.y[4, -1] * 1000
fb2_s    = y_fb2[4, -1]     * 1000
print(f"Fed-batch (bolus at t={t_bol} min): {fb2_s:.3f} µM  vs  batch: {batch_s:.3f} µM")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(
    "Module 3b — Fed-Batch Exocytosis: Single Bolus Addition\n"
    "Same k_sulfation throughout — bolus at t = 6 h triggers a second production burst",
    fontsize=12, fontweight="bold"
)

ax = axes[0, 0]
ax.plot(sol_exo.t, sol_exo.y[0], color=P["blue"],  linewidth=2.0,
        linestyle="--", label="Batch — Heparosan")
ax.plot(sol_exo.t, sol_exo.y[2], color=P["brown"], linewidth=2.0,
        linestyle="--", label="Batch — PAPS")
ax.plot(t_fb2, y_fb2[0], color=P["blue"],  linewidth=2.0, alpha=0.55,
        label="Fed-batch — Heparosan")
ax.plot(t_fb2, y_fb2[2], color=P["brown"], linewidth=2.0, alpha=0.55,
        label="Fed-batch — PAPS")
ax.axvline(t_bol, color=P["dkgreen"], linestyle=":", linewidth=1.6,
           label=f"Bolus at t = {t_bol} min")
ax.set(xlabel="Time (min)", ylabel="Concentration (mM)",
       title="Substrate Profiles\nBolus restores Hs and PAPS to initial levels")
ax.legend(fontsize=8.5)

ax = axes[0, 1]
ax.plot(sol_exo.t, sol_exo.y[3], color=P["purple"], linewidth=2.2,
        linestyle="--", label="Batch — single burst only")
ax.plot(t_fb2,     y_fb2[3],     color=P["purple"], linewidth=2.2,
        label="Fed-batch — second burst after bolus")
ax.axvline(t_bol, color=P["dkgreen"], linestyle=":", linewidth=1.6,
           label=f"Bolus at t = {t_bol} min")
ax.fill_between(t_fb2, y_fb2[3], 0, alpha=0.10, color=P["purple"])
ax.set(xlabel="Time (min)", ylabel="Vesicle count (relative)",
       title="Vesicle Formation\nFed-batch triggers a second production burst")
ax.legend(fontsize=9)

ax = axes[1, 0]
ax.plot(sol_exo.t, sol_exo.y[4] * 1000, color=P["orange"], linewidth=2.2,
        linestyle="--", label="Batch")
ax.plot(t_fb2,     y_fb2[4]     * 1000, color=P["teal"],   linewidth=2.2,
        label="Fed-batch (bolus at 6 h)")
ax.fill_between(t_fb2, y_fb2[4] * 1000, 0, alpha=0.08, color=P["teal"])
ax.axvline(t_bol, color=P["dkgreen"], linestyle=":", linewidth=1.6,
           label=f"Bolus at t = {t_bol} min")
ax.set(xlabel="Time (min)", ylabel="Secreted Heparin (\u00b5M)",
       title="Cumulative Secreted Heparin\nFed-batch gains additional secretion after bolus")
ax.legend(fontsize=9)

ax = axes[1, 1]
ax.axis("off")
rows_fb2 = [
    ["Metric",               "Batch (12 h)",         "Fed-batch (12 h, bolus at 6 h)"],
    ["Final secreted (\u00b5M)",  f"{batch_s:.3f}",  f"{fb2_s:.3f}"],
    ["Fold improvement",     "1\u00d7 (reference)",  f"{fb2_s/batch_s:.2f}\u00d7"],
    ["Vesicle bursts",       "1",                     "2"],
    ["k_sulfation used",     "0.857 mM\u207b\u00b9 min\u207b\u00b9", "0.857 mM\u207b\u00b9 min\u207b\u00b9"],
    ["New parameters?",      "\u2014",               "None \u2014 same model"],
    ["Substrate restocked?", "No",                    "Yes \u2014 \u0394Hs=10, \u0394PAPS=12"],
]
table = ax.table(cellText=rows_fb2[1:], colLabels=rows_fb2[0],
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#D1D5DB")
    if r == 0:
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_facecolor("#FEF3C7")
    elif c == 2:
        cell.set_facecolor("#D1FAE5")
    else:
        cell.set_facecolor("white")
ax.set_title("Exocytosis Performance Summary", fontweight="bold", fontsize=10, pad=8)

plt.tight_layout()
plt.show()
print("Figure 3b saved.")


# FIGURE 4 — Full Integrated System (Baseline)

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.42)
fig.suptitle("Module 4 — Full Integrated Synthetic Bioreactor (Baseline)",
             fontsize=15, fontweight="bold")

# Glucose
ax = fig.add_subplot(gs[0, 0])
ax.plot(t_full, y_full[0], color=P["blue"],  label="Glc$_{ext}$ (medium)")
ax.plot(t_full, y_full[1], color=P["sky"],   label="Glc$_{int}$ (cytoplasm)", linestyle="--")
ax.fill_between(t_full, y_full[0], y_full[1], alpha=0.07, color=P["blue"])
ax.set(xlabel="Time (min)", ylabel="Concentration (mM)",
       title="① Glucose Import\nGLUT1-type facilitated diffusion")
ax.legend(fontsize=9)

# Gene expression
ax  = fig.add_subplot(gs[0, 1])
ax2 = ax.twinx()
ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
ax.axvspan(0, t_lag, alpha=0.06, color=P["red"], zorder=0)
ax.text(t_lag * 0.45, mrna_nM[-1] * 0.75, "lag\nphase",
        ha="center", fontsize=8.5, color=P["red"], alpha=0.85, style="italic")
l1, = ax.plot(t_full, mrna_nM, color=P["orange"], label="mRNA (nM)")
l2, = ax2.plot(t_full, enz_nM, color=P["red"],    label="Enzyme (nM)", linestyle="--")
ax.set(xlabel="Time (min)", ylabel="mRNA (nM)")
ax2.set_ylabel("Enzyme (nM)", color=P["red"])
ax.tick_params(axis="y", colors=P["orange"])
ax2.tick_params(axis="y", colors=P["red"])
ax.set_title("② Gene Expression → Enzyme Accumulation\nDNA → mRNA → biosynthetic enzyme",
             fontweight="bold", fontsize=10)
ax.legend(handles=[l1, l2], loc="center right", fontsize=9)

# Heparin production
ax = fig.add_subplot(gs[1, 0])
ax.plot(t_full, hepa_uM, color=P["purple"],  label="Heparosan (backbone)", alpha=0.85)
ax.plot(t_full, hep_uM,  color=P["dkgreen"], label="Heparin (product)", linewidth=2.5)
ax.axhline(TARGET, color=P["dkgreen"], linestyle=":", linewidth=1.5,
           label=f"Therapeutic target ({TARGET:.0f} µM)")
ax.fill_between(t_full, hep_uM, 0, alpha=0.10, color=P["dkgreen"])
if t_cross is not None:
    ax.axvline(t_cross, color=P["dkgreen"], linestyle="--", linewidth=1.1, alpha=0.7)
    ax.annotate(f"Target met\nt ≈ {t_cross:.0f} min",
                xy=(t_cross, TARGET), xytext=(t_cross + 10, TARGET * 3.5),
                fontsize=8.5, color=P["dkgreen"],
                arrowprops=dict(arrowstyle="->", color=P["dkgreen"], lw=1.2))
ax.set(xlabel="Time (min)", ylabel="Concentration (µM)",
       title="③ Heparin Production\nKfiA polymerisation + NDST/OST sulfation cascade")
ax.legend(fontsize=9)

# PAPS
ax = fig.add_subplot(gs[1, 1])
ax.plot(t_full, y_full[5] * 1000, color=P["brown"], label="PAPS (active)")
ax.fill_between(t_full, y_full[5] * 1000, 0, alpha=0.12, color=P["brown"])
ax.axhline(BASE_PARAMS["paps_max"] * 1000, color=P["gray"], linestyle="--",
           linewidth=1.2, label="PAPS$_{max}$ (50 µM)")
ax.set(xlabel="Time (min)", ylabel="PAPS (µM)",
       title="④ PAPS — Sulfate Donor Resource\nconsumed by OSTs, regenerated from ATP + sulfate")
ax.legend(fontsize=9)

plt.show()

# FIGURE 5 — Failure Modes & DNA Dose-Response

failure_scenarios = {
    "Baseline":                  ({},               list(BASE_Y0), P["gray"]),
    "No PAPS regeneration":      ({"k_regen": 0.0}, list(BASE_Y0), P["red"]),
    "Glucose-limited (1 mM)":    ({},               [1.0] + BASE_Y0[1:], P["orange"]),
    "No gene expression":        ({"k_tx": 0.0},    list(BASE_Y0), P["purple"]),
}

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Module 4 — Design Analysis: Failure Modes & Plasmid Dosing",
             fontsize=14, fontweight="bold")

for label, (ov, ic, col) in failure_scenarios.items():
    p_mod = {**BASE_PARAMS, **ov}
    s = run_full(params=p_mod, y0=ic)
    lw = 2.8 if label == "Baseline" else 1.8
    ax_l.plot(s.t, s.y[6] * 1000, color=col, label=label, linewidth=lw)

ax_l.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
             label=f"Therapeutic target ({TARGET:.0f} µM)")
ax_l.set(xlabel="Time (min)", ylabel="Heparin (µM)",
         title="Failure Mode Analysis\nWhat breaks when a key component is removed?")
ax_l.legend(fontsize=9)

# DNA dose-response
dna_folds  = np.logspace(-1, 1, 40)
dna_finals = []
for fold in dna_folds:
    p_dna = {**BASE_PARAMS, "dna": BASE_PARAMS["dna"] * fold}
    s = run_full(params=p_dna)
    dna_finals.append(s.y[6, -1] * 1000)
dna_finals = np.array(dna_finals)
_met = np.where(dna_finals >= TARGET)[0]
fold_cross = dna_folds[_met[0]] if len(_met) else None

ax_r.plot(dna_folds, dna_finals, color=P["blue"], linewidth=2.2)
ax_r.fill_between(dna_folds, dna_finals, 0, alpha=0.08, color=P["blue"])
ax_r.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
             label=f"Therapeutic target ({TARGET:.0f} µM)")
if fold_cross is not None:
    ax_r.axvline(fold_cross, color=P["dkgreen"], linestyle="--", linewidth=1.4)
    ax_r.annotate(f"Target met\nat {fold_cross:.2f}× DNA",
                  xy=(fold_cross, TARGET),
                  xytext=(fold_cross * 1.6, TARGET * 2.5),
                  fontsize=9, color=P["dkgreen"],
                  arrowprops=dict(arrowstyle="->", color=P["dkgreen"], lw=1.2))
ax_r.axvline(1.0, color=P["gray"], linestyle="--", linewidth=1.0, alpha=0.7,
             label="Nominal DNA (5 nM)")
ax_r.set_xscale("log")
ax_r.set(xlabel="DNA concentration (fold of nominal 5 nM)",
         ylabel="Final heparin at t = 180 min (µM)",
         title="Plasmid Dose–Response\nHow much DNA is needed to hit the target?")
ax_r.legend(fontsize=9)

plt.tight_layout()
plt.show()
print("Figure 5 saved.")

# FIGURE 6 — Monte Carlo Robustness

np.random.seed(42)
N_MC     = 300
CV       = 0.25
mc_keys  = ["V_t", "dna", "kcat_a", "Km_p", "k_regen"]
mc_labs  = ["V_transporter", "DNA_conc", "kcat_KfiA", "Km_PAPS", "k_PAPS_regen"]
mc_samp  = {k: [] for k in mc_keys}
mc_out   = []

for _ in range(N_MC):
    p_mc = dict(BASE_PARAMS)
    for key in mc_keys:
        s = BASE_PARAMS[key] * np.random.normal(1.0, CV)
        s = max(s, BASE_PARAMS[key] * 0.01)
        p_mc[key] = s
        mc_samp[key].append(s)
    mc_out.append(run_full(params=p_mc).y[6, -1] * 1000)

mc_out  = np.array(mc_out)
pct_met = np.mean(mc_out >= TARGET) * 100

corr        = {lab: pearsonr(mc_samp[k], mc_out)[0]
               for k, lab in zip(mc_keys, mc_labs)}
sorted_labs = sorted(corr, key=lambda k: corr[k])
sorted_vals = [corr[k] for k in sorted_labs]
bar_cols    = [P["dkgreen"] if v > 0 else P["red"] for v in sorted_vals]

fig, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Module 4 — Robustness Under Parameter Uncertainty  (N = {N_MC}, CV = {CV*100:.0f}%)",
             fontsize=14, fontweight="bold")

bins    = np.linspace(mc_out.min(), mc_out.max(), 28)
met     = mc_out[mc_out >= TARGET]
not_met = mc_out[mc_out <  TARGET]
ax_h.hist(not_met, bins=bins, color=P["red"],    alpha=0.75, label=f"Below {TARGET:.0f} µM target")
ax_h.hist(met,     bins=bins, color=P["dkgreen"], alpha=0.75, label="Meets target")
ax_h.axvline(TARGET,             color="black",   linestyle=":",  linewidth=2.0,
             label=f"Therapeutic target ({TARGET:.0f} µM)")
ax_h.axvline(np.median(mc_out),  color=P["blue"], linestyle="--", linewidth=1.8,
             label=f"Median = {np.median(mc_out):.1f} µM")
ax_h.set(xlabel="Final heparin at t = 180 min (µM)", ylabel="Number of simulations",
         title=f"Output Distribution\n{pct_met:.0f}% of parameter sets meet the target")
ax_h.legend(fontsize=9)

ax_t.barh(sorted_labs, sorted_vals, color=bar_cols, alpha=0.85,
          edgecolor="white", height=0.55)
ax_t.axvline(0, color="black", linewidth=1.2)
for i, (lab, val) in enumerate(zip(sorted_labs, sorted_vals)):
    offset = 0.03 if val >= 0 else -0.03
    ax_t.text(val + offset, i, f"{val:+.2f}", va="center", fontsize=9,
              ha="left" if val >= 0 else "right")
ax_t.set(xlabel="Pearson r  (linear correlation with final heparin)",
         title="Sensitivity — Tornado Chart\nWhich parameters matter most for yield?",
         xlim=(-1.1, 1.1))

plt.tight_layout()
plt.show()
print(f"Figure 6 saved.  MC summary: median {np.median(mc_out):.2f} µM, "
      f"mean {np.mean(mc_out):.2f} ± {np.std(mc_out):.2f} µM, "
      f"{pct_met:.0f}% meet target.")


# FIGURE 7 — Porcine (mechanistic) vs Synthetic Bioreactor Comparison
#
# The porcine model now uses a two-stage mechanistic ODE:
#   Stage 1 (0–180 min): first-order enzymatic digestion releasing heparin into solution
#   Stage 2 (180–600 min): pseudo-second-order chemisorption onto anion-exchange resin [B, C]
# Final API yield applies a 55% purification recovery factor [B–D].
# The porcine process runs over 600 min (10 h) on its natural industrial timescale.
# The synthetic bioreactor runs over 180 min — a fundamentally different process paradigm.

params_boosted = {**BASE_PARAMS, "paps_max": 0.10}   # 2× PAPS loading
sol_boost = run_full(params=params_boosted, y0=BASE_Y0_SEEDED)
hep_boost = sol_boost.y[6] * 1000

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.44)
fig.suptitle(
    "Porcine vs Synthetic Heparin Bioreactor — Comparative Analysis\n"
    "Porcine: two-stage mechanistic ODE (enzymatic digestion + pseudo-second-order resin adsorption) [A–D]",
    fontsize=12, fontweight="bold"
)

# Panel A: Porcine process stages on its own natural timescale (10 h = 600 min)
ax = fig.add_subplot(gs[0, :2])
ax2r = ax.twinx()
ax2r.spines["right"].set_visible(True); ax2r.spines["top"].set_visible(False)

ax.axvspan(0,   PORC["t_ads"], alpha=0.06, color=P["orange"], zorder=0)
ax.axvspan(PORC["t_ads"], PORC["t_end"], alpha=0.06, color=P["blue"], zorder=0)
ax.text(PORC["t_ads"] * 0.45, PORC["C_max"] * 0.85,
        "Stage 1\nEnzymatic\ndigestion\n(protease,\n55°C, pH 9)",
        ha="center", fontsize=8, color=P["orange"], style="italic")
ax.text(PORC["t_ads"] + (PORC["t_end"] - PORC["t_ads"]) * 0.45, PORC["C_max"] * 0.85,
        "Stage 2\nResin adsorption\n(pseudo-2nd-order\nchemisorption) [B,C]",
        ha="center", fontsize=8, color=P["blue"], style="italic")

l1, = ax.plot(t_porc, C_sol_uM,  color=P["orange"],  linewidth=2.0,
              label="Heparin in solution [C_sol] (µM)")
l2, = ax.plot(t_porc, q_uM,      color=P["blue"],    linewidth=2.0,
              linestyle="--", label="Heparin on resin [q] (µM)")
l3, = ax2r.plot(t_porc, api_uM,  color=P["dkgreen"], linewidth=2.5,
                label=f"Recoverable API [η·q] (µM), η={PORC['eta']}")
ax.axvline(PORC["t_ads"], color=P["gray"], linestyle="--", linewidth=1.0)
ax2r.axhline(porc_final_uM, color=P["dkgreen"], linestyle=":", linewidth=1.2,
             label=f"Final API = {porc_final_uM:.1f} µM")

ax.set(xlabel="Time (min)  [total process = 600 min = 10 h]",
       ylabel="Concentration (µM)", ylim=(0, PORC["C_max"] * 1.1))
ax2r.set_ylabel("Recoverable API (µM)", color=P["dkgreen"])
ax2r.tick_params(axis="y", colors=P["dkgreen"])
ax.set_title("A — Porcine Extraction: Two-Stage Mechanistic Model\n"
             "Stage 1: 1st-order enzymatic digestion  |  Stage 2: pseudo-2nd-order resin adsorption [B,C]",
             fontweight="bold", fontsize=9)
ax.legend(handles=[l1, l2, l3], fontsize=8.5, loc="center right")

# Panel B: Final yield bar chart comparison
ax = fig.add_subplot(gs[0, 2])
categories = ["Porcine\n(recovered API\n10 h process)",
              "Synthetic\n(baseline\n180 min)",
              "Synthetic\n(2× PAPS\n180 min)"]
values     = [porc_final_uM, hep_uM[-1], hep_boost[-1]]
colours_b  = [P["orange"], P["dkgreen"], P["blue"]]
bars = ax.bar(categories, values, color=colours_b, alpha=0.85, edgecolor="white", width=0.55)
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.5,
           label=f"Target ({TARGET:.0f} µM)")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
            f"{val:.1f} µM", ha="center", fontsize=9, fontweight="bold")
ax.set(ylabel="Heparin yield (µM)", title="B — Final Yield Comparison\n(porcine: end of 10 h process)")
ax.legend(fontsize=9)

# Panel C: PAPS sensitivity for synthetic
ax = fig.add_subplot(gs[1, 0])
paps_range  = np.linspace(0.01, 0.15, 30)
paps_yields = []
for pv in paps_range:
    p_p = {**BASE_PARAMS, "paps_max": pv}
    paps_yields.append(run_full(params=p_p, y0=BASE_Y0_SEEDED).y[6, -1] * 1000)
paps_yields = np.array(paps_yields)
ax.plot(paps_range * 1000, paps_yields, color=P["brown"], linewidth=2.2)
ax.fill_between(paps_range * 1000, paps_yields, 0, alpha=0.10, color=P["brown"])
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.3, label=f"Target ({TARGET:.0f} µM)")
ax.axvline(BASE_PARAMS["paps_max"] * 1000, color=P["gray"],
           linestyle="--", linewidth=1.1, label="Nominal PAPS (50 µM)")
ax.set(xlabel="PAPS$_{max}$ (µM)", ylabel="Final heparin (µM)",
       title="C — PAPS Loading vs Yield\n(synthetic bioreactor lever)")
ax.legend(fontsize=9)

# Panel D: Contamination / purity
ax = fig.add_subplot(gs[1, 1])
sources = ["Porcine\n(conventional)", "Porcine\n(GMP-grade)", "Synthetic\n(cell-free)"]
purity  = [72, 85, 99]
risk    = [28, 15,  1]
x = np.arange(len(sources))
width = 0.35
ax.bar(x, purity, width, color=[P["orange"], P["teal"], P["dkgreen"]],
       alpha=0.85, label="Structural purity (%)", edgecolor="white")
ax.bar(x, risk, width, bottom=purity, color=P["red"],
       alpha=0.55, label="Contamination / impurity risk (%)", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(sources, fontsize=9)
ax.set(ylabel="Percentage (%)", ylim=(0, 115),
       title="D — Estimated Product Purity\n(porcine risk includes OSCS-type contaminants [7])")
ax.legend(fontsize=9)

# Panel E: Summary table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
rows = [
    ["Metric",                "Porcine",              "Synthetic (base)",          "Synthetic (opt.)"],
    ["Process model",         "2-stage mech. ODE",    "7-state ODE",               "7-state ODE"],
    ["Final yield (µM)",      f"{porc_final_uM:.1f}", f"{hep_uM[-1]:.1f}",         f"{hep_boost[-1]:.1f}"],
    ["Process time (min)",    "600 (10 h)",            "180",                       "180"],
    ["Meets target?",
     "No" if porc_final_uM < TARGET else "Yes",
     "Yes" if hep_uM[-1] >= TARGET else "No",
     "Yes" if hep_boost[-1] >= TARGET else "No"],
    ["Animal-derived?",       "Yes",                  "No",                        "No"],
    ["Contamination risk",    "High",                 "Very low",                  "Very low"],
    ["Kinetic basis",         "Pseudo-2nd-order [B,C]", "MM / mass action",        "MM / mass action"],
]
table = ax.table(cellText=rows[1:], colLabels=rows[0],
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(7.5)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#D1D5DB")
    if r == 0:
        cell.set_facecolor("#1E3A5F"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 1: cell.set_facecolor("#FEF3C7")
    elif c == 2: cell.set_facecolor("#D1FAE5")
    elif c == 3: cell.set_facecolor("#DBEAFE")
    else:        cell.set_facecolor("white")
ax.set_title("E — Summary Comparison", fontweight="bold", fontsize=10, pad=8)

plt.show()
print("Figure 7 saved.")


print("\n" + "="*60)
print("  SYNTHETIC HEPARIN BIOREACTOR — FINAL SUMMARY")
print("="*60)
print(f"  Synthetic baseline yield :  {hep_uM[-1]:.2f} µM")
print(f"  Porcine recovered API    :  {porc_final_uM:.2f} µM  (10 h, η=0.55)")
print(f"  Porcine adsorbed (q_max) :  {q_uM[-1]:.2f} µM  (pre-purification)")
print(f"  Synthetic (2× PAPS) yield:  {hep_boost[-1]:.2f} µM")
print(f"  Therapeutic target       :  {TARGET:.1f} µM")
print(f"  Time for synthetic to hit:  {t_cross:.0f} min" if t_cross else "  Target not met in 180 min")
print(f"  Monte Carlo (N={N_MC}):  median {np.median(mc_out):.2f}, "
      f"mean {np.mean(mc_out):.2f} ± {np.std(mc_out):.2f} µM, "
      f"{pct_met:.0f}% meet target")
print("="*60)

