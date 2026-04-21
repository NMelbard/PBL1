"""
Consolidated Heparin Synthetic-Cell Bioreactor Model
=====================================================
Integrates:
  Module 1 – IVTT Gene Expression (mRNA → Enzyme)
  Module 2 – Glucose Import & Precursor Synthesis (detailed pathway)
  Module 3 – Heparosan Modification & Sulfation (NDST → C5-Epi → 2OST → 6OST → 3OST)
  Module 4 – Cofactor / Energy Cycling (PAPS/PAP, ATP/ADP, NAD/NADH)
  Module 5 – Vesicle-mediated Exocytosis (packaging → secretion)
  Module 6 – Membrane Diffusion & Bioreactor Mass Balance (batch vs fed-batch)

Produces 6 publication-quality figures + economic / scale-up analysis.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.0,
    "axes.grid":         True,
    "grid.alpha":        0.20,
    "grid.linestyle":    "--",
    "lines.linewidth":   2.0,
    "legend.frameon":    False,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

PAL = {
    "blue":    "#3B82F6", "sky":     "#93C5FD", "orange":  "#F97316",
    "green":   "#22C55E", "dkgreen": "#15803D", "purple":  "#A855F7",
    "red":     "#EF4444", "brown":   "#92400E", "gray":    "#6B7280",
    "teal":    "#14B8A6", "pink":    "#EC4899", "indigo":  "#6366F1",
    "amber":   "#F59E0B", "slate":   "#475569",
}

# ═══════════════════════════════════════════════════════════════════════
# KINETIC HELPERS
# ═══════════════════════════════════════════════════════════════════════
def mm1(Vmax, S, Km):
    return Vmax * S / (Km + S + 1e-12)

def mm2(Vmax, A, B, KmA, KmB):
    d = KmA * KmB + KmB * A + KmA * B + A * B + 1e-12
    return Vmax * A * B / d

def v_packaging(H, K, Vmax, n):
    return Vmax * H**n / (K**n + H**n + 1e-12)

def ca_pulse(t, pulse_times, A=1.0, sigma=0.1):
    return sum(A * np.exp(-((t - ti)**2) / (2 * sigma**2)) for ti in pulse_times)

def k_exo(C, K, kmax, n):
    return kmax * C**n / (K**n + C**n + 1e-12)

def membrane_flux(D, C_out, C_in, L):
    return D * (C_out - C_in) / L

# ═══════════════════════════════════════════════════════════════════════
# BIOREACTOR GEOMETRY
# ═══════════════════════════════════════════════════════════════════════
def default_geometry(n_cells=5e10, batch_L=1.0):
    r = 10e-6                        # cell radius (m)
    L = 5e-9                         # membrane thickness (m)
    A = 4 * np.pi * r**2
    Vc = (4/3) * np.pi * r**3
    V_batch = batch_L * 1e-3         # m³
    V_out = max(V_batch - n_cells * Vc, 1e-9)
    return dict(
        r=r, L=L, A=A, Vc=Vc, n_cells=n_cells,
        V_out=V_out, V_batch_L=batch_L,
        D_glucose=6e-8,              # m²/s  (from bionumbers)
    )

# ═══════════════════════════════════════════════════════════════════════
# FULL PATHWAY ODE  (35 intracellular species)
# ═══════════════════════════════════════════════════════════════════════
PATHWAY_LABELS = [
    "HNAc","HNS","HEpi","H2S","H6S","H3S",       # heparin intermediates
    "PAPS","PAP",                                   # sulfate donor cycle
    "Glucose","G6P","F6P","G1P",                    # glycolysis branch
    "UDP_Glc","UDP_GlcA",                           # UDP-sugar branch A
    "GlcN6P","GlcN1P","GlcNAc1P","UDP_GlcNAc",    # UDP-sugar branch B
    "Glutamine","Glutamate","NH3",                  # nitrogen
    "AcetylCoA","CoASH",                            # acetyl pool
    "UTP","UDP","UMP",                              # uridylate pool
    "ATP","ADP","AMP",                              # adenylate pool
    "PPi","Pi",                                     # phosphate
    "APS","Sulfite",                                # sulfate assimilation
    "NADH","NAD",                                   # redox
]
N_PATH = len(PATHWAY_LABELS)  # 35

# Extra state indices
IDX = dict(R=35, Ca=36, V=37, S=38, G_out=39, mRNA=40, Protein=41)
N_TOTAL = 42

ALL_LABELS = PATHWAY_LABELS + ["R_reserve","Ca","V_release","S_secreted",
                                "G_out","mRNA","Protein"]

def default_params():
    return {
        # KfiA / KfiC polymerization
        "Vmax_kfiA": 1.0,  "Km_kfiA": 0.5,  "Km_kfiC": 0.5,
        # Sulfotransferases
        "Vmax_NS": 1.2, "Km_HNAc_NS": 0.8, "Km_PAPS_NS": 0.5,
        "Vmax_Epi": 0.9, "Km_Epi": 0.6,
        "Vmax_2OST": 0.8, "Km_HEpi_2OST": 0.7, "Km_PAPS_2OST": 0.5,
        "Vmax_6OST": 0.7, "Km_H2S_6OST": 0.7, "Km_PAPS_6OST": 0.5,
        "Vmax_3OST": 0.4, "Km_H6S_3OST": 0.6, "Km_PAPS_3OST": 0.5,
        "Ki_PAP": 1.0,
        "inhibitor_2OST": 0.0, "Ki_2OST": 1e12,
        "inhibitor_3OST": 0.0, "Ki_3OST": 1e12,
        # Hexokinase
        "Vmax_hex": 2.0, "Km_Glucose_hex": 1.0, "Km_ATP_hex": 0.5,
        # PGI (reversible)
        "Vmax_PGI_f": 2.0, "Km_G6P_PGI": 0.5,
        "Vmax_PGI_r": 1.5, "Km_F6P_PGI": 0.5,
        # PGM (reversible)
        "Vmax_PGM_f": 1.5, "Km_G6P_PGM": 0.5,
        "Vmax_PGM_r": 1.0, "Km_G1P_PGM": 0.5,
        # GlmS, GlmM, GlmU
        "Vmax_GlmS": 1.2, "Km_F6P_GlmS": 0.5, "Km_Gln_GlmS": 0.5,
        "Vmax_GlmM": 1.0, "Km_GlcN6P_GlmM": 0.5,
        "Vmax_GlmU_acetyl": 1.0, "Km_GlcN1P_GlmU_acetyl": 0.5, "Km_AcetylCoA_GlmU_acetyl": 0.5,
        "Vmax_GlmU_UTP": 1.0, "Km_GlcNAc1P_GlmU_UTP": 0.5, "Km_UTP_GlmU_UTP": 0.5,
        # GalU, KfiD
        "Vmax_GalU": 1.0, "Km_G1P_GalU": 0.5, "Km_UTP_GalU": 0.5,
        "Vmax_KfiD": 0.8, "Km_UDP_Glc_KfiD": 0.5, "Km_NAD_KfiD": 0.5,
        # Recycling / ancillary
        "k_gln_synth": 0.01, "k_acs": 0.01, "Acetate_pool": 10.0,
        "k_ndk_f": 0.01, "k_ndk_r": 0.005, "k_umpk": 0.01, "k_ppase": 0.05,
        # PAPS regeneration (sulfate assimilation)
        "Vmax_ATPS": 0.8, "Km_Sulfite_ATPS": 0.5, "Km_ATP_ATPS": 0.5,
        "Vmax_APSK": 0.8, "Km_APS_APSK": 0.5, "Km_ATP_APSK": 0.5,
        "k_PAPase": 0.05,
        # Energy maintenance
        "k_AMP_recharge": 0.005, "k_ATP_regen": 0.3, "k_NAD_regen": 0.2,
    }

def default_exo():
    return dict(
        Kpack=0.1, Kexo=0.001, Vmax_pack=0.1, kmax=5.0,
        n1=1, n2=2, kdeg=0.05, krel=0.3, kca=5.0,
        q=1e-3, A=1.0, sigma=0.1,
    )

def default_gene():
    return dict(k_R=20.0, gamma_R=0.49, k_P=2.28, gamma_P=0.00057)

def default_scale():
    return dict(enabled=True, beta0=0.85, beta1=0.15, K_P=100.0,
                keys=("Vmax_kfiA","Vmax_hex","Vmax_NS"))

def default_y0(geo):
    path0 = [
        0,0,0,0,0,0,      # heparin intermediates
        20.0, 0.0,          # PAPS, PAP
        0.5, 0,0,0,         # Glc_in, G6P, F6P, G1P
        0,0,                # UDP_Glc, UDP_GlcA
        0,0,0,0,            # GlcN6P..UDP_GlcNAc
        10,5,10,            # Gln, Glu, NH3
        10,5,               # AcCoA, CoASH
        15,5,5,             # UTP, UDP, UMP
        40,10,2,            # ATP, ADP, AMP
        0,10,               # PPi, Pi
        5,10,               # APS, Sulfite
        5,20,               # NADH, NAD
    ]
    return path0 + [90.0, 0.0, 10.0, 0.0,  # R, Ca, V, S
                    5.0,                     # G_out (mM)
                    0.0, 0.0]               # mRNA, Protein


def pathway_rhs(t, y, p):
    """35-species intracellular pathway."""
    (HNAc, HNS, HEpi, H2S, H6S, H3S,
     PAPS, PAP,
     Glc, G6P, F6P, G1P,
     UDP_Glc, UDP_GlcA,
     GlcN6P, GlcN1P, GlcNAc1P, UDP_GlcNAc,
     Gln, Glu, NH3,
     AcCoA, CoASH,
     UTP, UDP, UMP,
     ATP, ADP, AMP,
     PPi, Pi,
     APS, Sulfite,
     NADH, NAD) = y

    # --- Glycolysis / sugar nucleotide synthesis ---
    v_hex = mm2(p["Vmax_hex"], Glc, ATP, p["Km_Glucose_hex"], p["Km_ATP_hex"])
    v_PGI_f = mm1(p["Vmax_PGI_f"], G6P, p["Km_G6P_PGI"])
    v_PGI_r = mm1(p["Vmax_PGI_r"], F6P, p["Km_F6P_PGI"])
    v_PGM_f = mm1(p["Vmax_PGM_f"], G6P, p["Km_G6P_PGM"])
    v_PGM_r = mm1(p["Vmax_PGM_r"], G1P, p["Km_G1P_PGM"])
    v_GlmS  = mm2(p["Vmax_GlmS"], F6P, Gln, p["Km_F6P_GlmS"], p["Km_Gln_GlmS"])
    v_GlmM  = mm1(p["Vmax_GlmM"], GlcN6P, p["Km_GlcN6P_GlmM"])
    v_GlmU_ac = mm2(p["Vmax_GlmU_acetyl"], GlcN1P, AcCoA,
                     p["Km_GlcN1P_GlmU_acetyl"], p["Km_AcetylCoA_GlmU_acetyl"])
    v_GlmU_utp = mm2(p["Vmax_GlmU_UTP"], GlcNAc1P, UTP,
                      p["Km_GlcNAc1P_GlmU_UTP"], p["Km_UTP_GlmU_UTP"])
    v_GalU = mm2(p["Vmax_GalU"], G1P, UTP, p["Km_G1P_GalU"], p["Km_UTP_GalU"])
    v_KfiD = mm2(p["Vmax_KfiD"], UDP_Glc, NAD, p["Km_UDP_Glc_KfiD"], p["Km_NAD_KfiD"])

    # --- Heparosan polymerization (KfiA + KfiC coupling) ---
    v_kfiA_base = mm1(p["Vmax_kfiA"], UDP_GlcNAc, p["Km_kfiA"])
    coupling = UDP_GlcA / (p["Km_kfiC"] + UDP_GlcA + 1e-12)
    v_poly = v_kfiA_base * coupling

    # --- Sulfation cascade with PAP product inhibition ---
    pap_inh = 1.0 / (1.0 + PAP / (p["Ki_PAP"] + 1e-12))
    vNS   = pap_inh * mm2(p["Vmax_NS"],   HNAc, PAPS, p["Km_HNAc_NS"],  p["Km_PAPS_NS"])
    vEpi  = mm1(p["Vmax_Epi"], HNS, p["Km_Epi"])
    inh2  = 1.0 / (1.0 + p["inhibitor_2OST"] / (p["Ki_2OST"] + 1e-12))
    v2OST = pap_inh * inh2 * mm2(p["Vmax_2OST"], HEpi, PAPS, p["Km_HEpi_2OST"], p["Km_PAPS_2OST"])
    v6OST = pap_inh * mm2(p["Vmax_6OST"], H2S, PAPS, p["Km_H2S_6OST"], p["Km_PAPS_6OST"])
    inh3  = 1.0 / (1.0 + p["inhibitor_3OST"] / (p["Ki_3OST"] + 1e-12))
    v3OST = pap_inh * inh3 * mm2(p["Vmax_3OST"], H6S, PAPS, p["Km_H6S_3OST"], p["Km_PAPS_3OST"])

    # --- Ancillary ---
    v_gln = p["k_gln_synth"] * Glu * NH3 * ATP
    v_acs = p["k_acs"] * p["Acetate_pool"] * CoASH * ATP
    v_ndk_f = p["k_ndk_f"] * ATP * UDP
    v_ndk_r = p["k_ndk_r"] * ADP * UTP
    v_umpk  = p["k_umpk"] * UMP * ATP
    v_ppase = p["k_ppase"] * PPi
    v_ATPS  = mm2(p["Vmax_ATPS"], Sulfite, ATP, p["Km_Sulfite_ATPS"], p["Km_ATP_ATPS"])
    v_APSK  = mm2(p["Vmax_APSK"], APS, ATP, p["Km_APS_APSK"], p["Km_ATP_APSK"])
    v_PAPase = p["k_PAPase"] * PAP
    v_AMP_re = p["k_AMP_recharge"] * AMP * ATP
    v_ATP_re = p["k_ATP_regen"] * ADP
    v_NAD_re = p["k_NAD_regen"] * NADH

    d = [0.0] * N_PATH
    d[0]  = v_poly - vNS                       # HNAc
    d[1]  = vNS - vEpi                          # HNS
    d[2]  = vEpi - v2OST                        # HEpi
    d[3]  = v2OST - v6OST                       # H2S
    d[4]  = v6OST - v3OST                       # H6S
    d[5]  = v3OST                               # H3S (mature heparin)
    d[6]  = -(vNS+v2OST+v6OST+v3OST) + v_APSK  # PAPS
    d[7]  = (vNS+v2OST+v6OST+v3OST) - v_PAPase # PAP
    d[8]  = -v_hex                              # Glucose_in
    d[9]  = v_hex - v_PGI_f + v_PGI_r - v_PGM_f + v_PGM_r  # G6P
    d[10] = v_PGI_f - v_PGI_r - v_GlmS         # F6P
    d[11] = v_PGM_f - v_PGM_r - v_GalU         # G1P
    d[12] = v_GalU - v_KfiD                     # UDP_Glc
    d[13] = v_KfiD - v_poly                     # UDP_GlcA
    d[14] = v_GlmS - v_GlmM                    # GlcN6P
    d[15] = v_GlmM - v_GlmU_ac                 # GlcN1P
    d[16] = v_GlmU_ac - v_GlmU_utp             # GlcNAc1P
    d[17] = v_GlmU_utp - v_poly                # UDP_GlcNAc
    d[18] = -v_GlmS + v_gln                    # Glutamine
    d[19] = v_GlmS - v_gln                     # Glutamate
    d[20] = -v_gln                              # NH3
    d[21] = -v_GlmU_ac + v_acs                 # AcetylCoA
    d[22] = v_GlmU_ac - v_acs                  # CoASH
    d[23] = -v_GalU - v_GlmU_utp + v_ndk_f - v_ndk_r  # UTP
    d[24] = -v_ndk_f + v_ndk_r + v_umpk        # UDP
    d[25] = -v_umpk                             # UMP
    d[26] = (-v_hex - v_gln - v_acs - v_ndk_f + v_ndk_r
             - v_umpk - v_ATPS - v_APSK - v_AMP_re + v_ATP_re)  # ATP
    d[27] = (v_hex + v_gln + v_ndk_f - v_ndk_r
             + v_umpk + v_APSK + 2*v_AMP_re - v_ATP_re)         # ADP
    d[28] = v_acs + v_PAPase - v_AMP_re         # AMP
    d[29] = v_GalU + v_GlmU_utp + v_acs + v_ATPS - v_ppase  # PPi
    d[30] = 2*v_ppase + v_gln + v_PAPase        # Pi
    d[31] = v_ATPS - v_APSK                     # APS
    d[32] = -v_ATPS                             # Sulfite
    d[33] = 2*v_KfiD - v_NAD_re                 # NADH
    d[34] = -2*v_KfiD + v_NAD_re                # NAD
    return d


def combined_ode(t, y, p, pulse_times, geo, exo, gene, scale, feed=None):
    """Full 42-state ODE: pathway + exocytosis + diffusion + gene expression."""
    py = np.maximum(np.array(y[:N_PATH], dtype=float), 0.0)
    R, Ca_val, V_val, S_val = [max(y[i],0) for i in (35,36,37,38)]
    G_out = max(y[39], 0.0)
    m, P = max(y[40], 0.0), max(y[41], 0.0)

    # Scale Vmax by protein level
    p_eff = dict(p)
    if scale.get("enabled"):
        f = scale["beta0"] + scale["beta1"] * P / (P + scale["K_P"] + 1e-12)
        for k in scale.get("keys", ()):
            if k in p_eff:
                p_eff[k] = p[k] * f

    dp = pathway_rhs(t, list(py), p_eff)
    H3S = py[5]

    # Exocytosis
    vp = v_packaging(H3S, exo["Kpack"], exo["Vmax_pack"], exo["n1"])
    ve = k_exo(Ca_val, exo["Kexo"], exo["kmax"], exo["n2"])
    dp[5] += -vp - exo["kdeg"] * H3S
    dR  = (1/exo["q"]) * vp - exo["krel"] * R
    dCa = ca_pulse(t, pulse_times, exo["A"], exo["sigma"]) - exo["kca"] * Ca_val
    dV  = exo["krel"] * R - ve * V_val
    dS  = exo["q"] * ve * V_val

    # Membrane diffusion (glucose)
    J = membrane_flux(geo["D_glucose"], G_out, py[8], geo["L"])
    dp[8] += 60.0 * (geo["A"] / geo["Vc"]) * J
    dG_out = 60.0 * (-(geo["n_cells"] * geo["A"] / geo["V_out"]) * J)

    # Continuous feed
    if feed is not None:
        F_L_min = feed["F_in_L_per_hr"] / 60.0
        V_out_L = geo["V_out"] * 1000.0
        dG_out += (F_L_min / V_out_L) * (feed["G_feed_mM"] - G_out)

    # Gene expression
    dm = gene["k_R"] - gene["gamma_R"] * m
    dP = gene["k_P"] * m - gene["gamma_P"] * P

    return dp + [dR, dCa, dV, dS, dG_out, dm, dP]


def simulate(p, geo, exo, gene, scale, t_end=120.0, pulse_every=5.0,
             feed=None, n_pts=400, y0=None):
    pulse_times = np.arange(5.0, t_end, pulse_every)
    if y0 is None:
        y0 = default_y0(geo)
    t_eval = np.linspace(0, t_end, n_pts)
    sol = solve_ivp(
        lambda t, y: combined_ode(t, y, p, pulse_times, geo, exo, gene, scale, feed),
        (0, t_end), y0, t_eval=t_eval, method="LSODA", atol=1e-8, rtol=1e-6,
    )
    return sol


# ═══════════════════════════════════════════════════════════════════════
# SIMPLE MODEL  (for comparison / validation — from super-simple.py)
# ═══════════════════════════════════════════════════════════════════════
def simple_ode(t, y, p):
    ge, gi, mrna, e, hs, paps, hep = [max(v,0) for v in y]
    v_imp  = p["V_t"] * ge / (p["Km_t"] + ge)
    v_tx   = p["k_tx"] * p["dna"]
    v_dm   = p["k_dm"] * mrna
    v_tl   = p["k_tl"] * mrna
    v_de   = p["k_de"] * e
    v_poly = p["kcat_a"] * e * gi / (p["Km_a"] + gi)
    den    = (p["Km_h"]*p["Km_p"] + p["Km_p"]*hs + p["Km_h"]*paps + hs*paps + 1e-12)
    v_sulf = p["kcat_m"] * e * hs * paps / den
    v_reg  = p["k_regen"] * (p["paps_max"] - paps)
    return [-v_imp, v_imp-v_poly, v_tx-v_dm, v_tl-v_de,
            v_poly-v_sulf, -v_sulf+v_reg, v_sulf]

simple_params = dict(
    V_t=0.10, Km_t=1.5, dna=5e-6, k_tx=0.10, k_dm=0.14,
    k_tl=0.06, k_de=0.004, kcat_a=200.0, Km_a=0.50,
    kcat_m=40.0, Km_h=0.05, Km_p=0.05,
    paps_max=0.05, k_regen=0.010,
)
simple_y0 = [10.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0]

def run_simple(p=None, y0=None, t_span=(0,180), n=600):
    if p is None: p = simple_params
    if y0 is None: y0 = simple_y0
    return solve_ivp(lambda t,y: simple_ode(t,y,p), t_span, y0,
                     t_eval=np.linspace(*t_span, n), method="LSODA",
                     rtol=1e-8, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# ECONOMIC / SCALE-UP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def economic_analysis(sol, geo, t_end_min):
    """
    Compute production rates, annual output, and cost estimates.
    Heparin MW ≈ 15,000 Da  →  1 mM = 15 g/L
    """
    MW_heparin = 15000  # g/mol
    S_final_mM = sol.y[IDX["S"], -1]                   # mM secreted heparin
    V_out_L    = geo["V_out"] * 1000.0
    n_cells    = geo["n_cells"]

    # Mass produced per batch
    mass_g_per_batch = S_final_mM * MW_heparin * V_out_L / 1e6  # g
    mass_mg = mass_g_per_batch * 1000

    # Batch time
    batch_hr = t_end_min / 60.0

    # Assume 80% uptime, ~330 operating days/yr
    batches_per_day = (24 * 0.80) / batch_hr
    batches_per_yr  = batches_per_day * 330
    annual_g        = mass_g_per_batch * batches_per_yr
    annual_kg       = annual_g / 1000

    # Global heparin demand ≈ 100,000 kg/yr (porcine-derived)
    global_demand_kg = 100_000
    pct_demand       = (annual_kg / global_demand_kg) * 100

    # Cost estimates (order-of-magnitude)
    # IVTT reagent: ~$50/mL reaction, but at industrial scale ~$5/L after optimization
    cost_ivtt_per_L    = 5.0    # $/L (optimistic industrial)
    cost_glucose_per_L = 0.10   # $/L (glucose + salts)
    cost_paps_per_L    = 2.0    # $/L (PAPS + ATP regeneration)
    overhead_per_batch = 50.0   # $/batch (labor, energy, QC)

    reagent_cost = (cost_ivtt_per_L + cost_glucose_per_L + cost_paps_per_L) * V_out_L
    total_batch_cost = reagent_cost + overhead_per_batch
    cost_per_g = total_batch_cost / max(mass_g_per_batch, 1e-12)

    # Porcine heparin cost ≈ $100-300/g (pharmaceutical grade)
    porcine_cost_per_g = 200.0

    return {
        "S_final_mM":       S_final_mM,
        "mass_mg_batch":    mass_mg,
        "mass_g_batch":     mass_g_per_batch,
        "batch_hr":         batch_hr,
        "batches_per_yr":   batches_per_yr,
        "annual_kg":        annual_kg,
        "pct_global_demand":pct_demand,
        "cost_per_batch":   total_batch_cost,
        "cost_per_g":       cost_per_g,
        "porcine_cost_g":   porcine_cost_per_g,
        "n_cells":          n_cells,
        "V_out_L":          V_out_L,
    }


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — SYSTEM OVERVIEW (Simple Model)
# ═══════════════════════════════════════════════════════════════════════
def figure1(save=True):
    sol = run_simple()
    t, y = sol.t, sol.y
    hep_uM  = y[6]*1e3
    hs_uM   = y[4]*1e3
    enz_nM  = y[3]*1e6
    mrna_nM = y[2]*1e6
    TARGET  = 10.0

    _c = np.where(hep_uM >= TARGET)[0]
    t_cross = t[_c[0]] if len(_c) else None
    _l = np.where(enz_nM >= 0.5*enz_nM[-1])[0]
    t_lag = t[_l[0]] if len(_l) else 40

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.40)
    fig.suptitle("Simplified Synthetic Cell — Heparin Biosynthesis Overview",
                 fontsize=14, fontweight="bold", y=0.98)

    # ① Glucose
    ax = fig.add_subplot(gs[0,0])
    ax.plot(t, y[0], color=PAL["blue"],  label="Glc$_{ext}$")
    ax.plot(t, y[1], color=PAL["sky"],   label="Glc$_{int}$", ls="--")
    ax.fill_between(t, y[0], y[1], alpha=0.06, color=PAL["blue"])
    ax.set(xlabel="Time (min)", ylabel="Concentration (mM)",
           title="① Glucose Import\nGLUT1 facilitated diffusion")
    ax.legend()

    # ② Gene expression
    ax = fig.add_subplot(gs[0,1])
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    ax.axvspan(0, t_lag, alpha=0.05, color=PAL["red"])
    ax.text(t_lag*0.4, mrna_nM[-1]*0.7, "lag\nphase", ha="center",
            fontsize=8, color=PAL["red"], alpha=0.8, style="italic")
    l1, = ax.plot(t, mrna_nM, color=PAL["orange"], label="mRNA")
    l2, = ax2.plot(t, enz_nM, color=PAL["red"], ls="--", label="Enzyme")
    ax.set(xlabel="Time (min)", ylabel="mRNA (nM)")
    ax2.set_ylabel("Enzyme (nM)", color=PAL["red"])
    ax.tick_params(axis="y", colors=PAL["orange"])
    ax2.tick_params(axis="y", colors=PAL["red"])
    ax.set_title("② Gene Expression\nDNA → mRNA → Enzyme")
    ax.legend(handles=[l1,l2], loc="center right")

    # ③ Heparin production
    ax = fig.add_subplot(gs[1,0])
    ax.plot(t, hs_uM, color=PAL["purple"], alpha=0.8, label="Heparosan")
    ax.plot(t, hep_uM, color=PAL["dkgreen"], lw=2.5, label="Heparin")
    ax.axhline(TARGET, color=PAL["dkgreen"], ls=":", lw=1.3, label=f"Target ({TARGET:.0f} µM)")
    ax.fill_between(t, hep_uM, 0, alpha=0.08, color=PAL["dkgreen"])
    if t_cross:
        ax.axvline(t_cross, color=PAL["dkgreen"], ls="--", lw=1, alpha=0.6)
        ax.annotate(f"target met\nt ≈ {t_cross:.0f} min", xy=(t_cross, TARGET),
                    xytext=(t_cross+12, TARGET*3.5), fontsize=8, color=PAL["dkgreen"],
                    arrowprops=dict(arrowstyle="->", color=PAL["dkgreen"], lw=1))
    ax.set(xlabel="Time (min)", ylabel="Concentration (µM)",
           title="③ Heparin Production\nKfiA polymerization + sulfation cascade")
    ax.legend()

    # ④ PAPS
    ax = fig.add_subplot(gs[1,1])
    ax.plot(t, y[5]*1e3, color=PAL["brown"], label="PAPS")
    ax.fill_between(t, y[5]*1e3, 0, alpha=0.10, color=PAL["brown"])
    ax.axhline(50, color=PAL["gray"], ls="--", lw=1, label="PAPS$_{max}$ (50 µM)")
    ax.set(xlabel="Time (min)", ylabel="PAPS (µM)",
           title="④ PAPS Cofactor Dynamics\nConsumed by sulfation, regenerated from ATP")
    ax.legend()

    if save: plt.savefig("fig1_system_overview.png")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — FAILURE MODES & DNA DOSE-RESPONSE
# ═══════════════════════════════════════════════════════════════════════
def figure2(save=True):
    T_SPAN = (0, 180)
    TARGET = 10.0

    scenarios = {
        "Baseline":              ({},                simple_y0, PAL["gray"]),
        "No PAPS regeneration":  ({"k_regen":0},     simple_y0, PAL["red"]),
        "Glucose-limited (1 mM)":({},[1]+simple_y0[1:], PAL["orange"]),
        "No gene expression":    ({"k_tx":0},        simple_y0, PAL["purple"]),
    }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Design Robustness Analysis", fontsize=13, fontweight="bold")

    for label, (ov, ic, col) in scenarios.items():
        p = {**simple_params, **ov}
        s = run_simple(p, list(ic), T_SPAN)
        lw = 2.8 if label == "Baseline" else 1.8
        axL.plot(s.t, s.y[6]*1e3, color=col, label=label, lw=lw)
    axL.axhline(TARGET, color="k", ls=":", lw=1.3, label=f"Target ({TARGET:.0f} µM)")
    axL.set(xlabel="Time (min)", ylabel="Heparin (µM)",
            title="Failure Mode Analysis\nEffect of removing key components")
    axL.legend(fontsize=8)

    # DNA dose-response
    folds = np.logspace(-1, 1, 40)
    finals = []
    for f in folds:
        p = {**simple_params, "dna": simple_params["dna"]*f}
        s = run_simple(p, simple_y0, T_SPAN)
        finals.append(s.y[6,-1]*1e3)
    finals = np.array(finals)

    axR.plot(folds, finals, color=PAL["blue"], lw=2.2)
    axR.fill_between(folds, finals, 0, alpha=0.06, color=PAL["blue"])
    axR.axhline(TARGET, color="k", ls=":", lw=1.3)
    _m = np.where(finals >= TARGET)[0]
    if len(_m):
        fc = folds[_m[0]]
        axR.axvline(fc, color=PAL["dkgreen"], ls="--", lw=1.3)
        axR.annotate(f"Target met at {fc:.2f}× DNA", xy=(fc, TARGET),
                     xytext=(fc*1.5, TARGET*2.5), fontsize=8, color=PAL["dkgreen"],
                     arrowprops=dict(arrowstyle="->", color=PAL["dkgreen"]))
    axR.set_xscale("log")
    axR.set(xlabel="DNA concentration (fold of 5 nM)", ylabel="Final heparin (µM)",
            title="Plasmid Dose–Response\nDNA loading vs. heparin yield at t=180 min")

    plt.tight_layout()
    if save: plt.savefig("fig2_failure_modes.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — MONTE CARLO SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════
def figure3(save=True):
    np.random.seed(42)
    N, CV, TARGET = 300, 0.25, 10.0
    T_SPAN = (0, 180)
    keys   = ["V_t","dna","kcat_a","Km_p","k_regen"]
    labels = ["V_transporter","DNA_conc","kcat_KfiA","Km_PAPS","k_PAPS_regen"]
    samples = {k: [] for k in keys}
    outputs = []

    for _ in range(N):
        pm = dict(simple_params)
        for k in keys:
            v = simple_params[k] * np.random.normal(1.0, CV)
            v = max(v, simple_params[k]*0.01)
            pm[k] = v
            samples[k].append(v)
        s = run_simple(pm, simple_y0, T_SPAN)
        outputs.append(s.y[6,-1]*1e3)

    outputs = np.array(outputs)
    pct_met = np.mean(outputs >= TARGET)*100

    corr = {l: pearsonr(samples[k], outputs)[0] for k, l in zip(keys, labels)}
    sl = sorted(corr, key=lambda x: corr[x])
    sv = [corr[k] for k in sl]
    bc = [PAL["dkgreen"] if v>0 else PAL["red"] for v in sv]

    fig, (axH, axT) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Robustness Under Parameter Uncertainty (N={N}, CV=±{CV*100:.0f}%)",
                 fontsize=13, fontweight="bold")

    met = outputs[outputs >= TARGET]
    nmet = outputs[outputs < TARGET]
    bins = np.linspace(outputs.min(), outputs.max(), 30)
    axH.hist(nmet, bins, color=PAL["red"], alpha=0.7, label=f"Below target")
    axH.hist(met,  bins, color=PAL["dkgreen"], alpha=0.7, label=f"Meets target")
    axH.axvline(TARGET, color="k", ls=":", lw=1.8)
    axH.axvline(np.median(outputs), color=PAL["blue"], ls="--", lw=1.5,
                label=f"Median = {np.median(outputs):.1f} µM")
    axH.set(xlabel="Final heparin (µM)", ylabel="Count",
            title=f"Output Distribution\n{pct_met:.0f}% of runs meet ≥{TARGET:.0f} µM target")
    axH.legend(fontsize=8)

    axT.barh(sl, sv, color=bc, alpha=0.85, edgecolor="white", height=0.55)
    axT.axvline(0, color="k", lw=1)
    for i, (l, v) in enumerate(zip(sl, sv)):
        axT.text(v + (0.03 if v>=0 else -0.03), i, f"{v:+.2f}",
                 va="center", fontsize=9, ha="left" if v>=0 else "right")
    axT.set(xlabel="Pearson r with final heparin",
            title="Sensitivity Tornado Chart\nWhich parameters most affect yield?")
    axT.set_xlim(-1.1, 1.1)

    plt.tight_layout()
    if save: plt.savefig("fig3_sensitivity.png")
    plt.close()

    print(f"\n{'='*55}")
    print(f"  Monte Carlo Summary  (N={N}, CV=±{CV*100:.0f}%)")
    print(f"{'='*55}")
    print(f"  Median:  {np.median(outputs):.2f} µM")
    print(f"  Mean:    {np.mean(outputs):.2f} ± {np.std(outputs):.2f} µM")
    print(f"  5–95th:  {np.percentile(outputs,5):.2f} – {np.percentile(outputs,95):.2f} µM")
    print(f"  Target met: {pct_met:.0f}%")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4 — FULL COMBINED MODEL (batch vs fed-batch)
# ═══════════════════════════════════════════════════════════════════════
def figure4(save=True):
    geo  = default_geometry()
    p    = default_params()
    exo  = default_exo()
    gene = default_gene()
    sc   = default_scale()
    t_end = 120.0

    # Batch (no feed)
    sol_nf = simulate(p, geo, exo, gene, sc, t_end=t_end)

    # Estimate glucose demand and set up feed
    t_arr = sol_nf.t
    G_out = sol_nf.y[IDX["G_out"]]
    mask = (t_arr >= 0) & (t_arr <= 20)
    slope = np.polyfit(t_arr[mask], G_out[mask], 1)[0]
    demand_mmol_hr = -slope * (geo["V_out"]*1000) * 60
    G_feed = 100.0
    F_in = max(demand_mmol_hr / G_feed, 0.001)
    feed = {"F_in_L_per_hr": F_in, "G_feed_mM": G_feed}

    sol_f = simulate(p, geo, exo, gene, sc, t_end=t_end, feed=feed)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Full Combined Model — Bioreactor Simulation",
                 fontsize=14, fontweight="bold", y=1.01)

    # (a) Heparin intermediates
    ax = axes[0,0]
    for i, (name, c) in enumerate(zip(
            ["HNAc","HNS","HEpi","H2S","H6S","H3S"],
            [PAL["blue"],PAL["sky"],PAL["orange"],PAL["amber"],PAL["purple"],PAL["dkgreen"]])):
        ax.plot(sol_nf.t, sol_nf.y[i], color=c, label=name, lw=1.5 if i<5 else 2.5)
    ax.set(xlabel="Time (min)", ylabel="Concentration (mM)",
           title="(a) Sulfation Cascade Intermediates")
    ax.legend(ncol=2, fontsize=7)

    # (b) Cytosolic H3S vs Secreted
    ax = axes[0,1]
    ax.plot(sol_nf.t, sol_nf.y[5], color=PAL["purple"], label="H3S (cytosolic)", lw=2)
    ax.plot(sol_nf.t, sol_nf.y[IDX["S"]], color=PAL["dkgreen"], label="Secreted heparin", lw=2.5)
    ax.set(xlabel="Time (min)", ylabel="Concentration (mM)",
           title="(b) Product: Cytosolic vs. Secreted Heparin")
    ax.legend()

    # (c) Glucose: batch vs fed-batch
    ax = axes[0,2]
    ax.plot(sol_nf.t, sol_nf.y[IDX["G_out"]], color=PAL["red"], lw=2, label="Batch (no feed)")
    ax.plot(sol_f.t, sol_f.y[IDX["G_out"]], color=PAL["dkgreen"], lw=2, label="Fed-batch")
    ax.plot(sol_nf.t, sol_nf.y[8], color=PAL["red"], lw=1.2, ls="--", alpha=0.6, label="G$_{in}$ (batch)")
    ax.set(xlabel="Time (min)", ylabel="Glucose (mM)",
           title="(c) Glucose: Batch vs. Fed-Batch")
    ax.legend(fontsize=7)

    # (d) Cofactors
    ax = axes[1,0]
    ax.plot(sol_nf.t, sol_nf.y[6], color=PAL["brown"], label="PAPS", lw=2)
    ax.plot(sol_nf.t, sol_nf.y[7], color=PAL["amber"], label="PAP", lw=1.5, ls="--")
    ax.plot(sol_nf.t, sol_nf.y[26], color=PAL["blue"], label="ATP", lw=1.5)
    ax.plot(sol_nf.t, sol_nf.y[27], color=PAL["sky"], label="ADP", lw=1.5, ls="--")
    ax.set(xlabel="Time (min)", ylabel="Concentration (mM)",
           title="(d) Cofactor & Energy Dynamics")
    ax.legend(ncol=2, fontsize=7)

    # (e) Vesicle compartments
    ax = axes[1,1]
    ax.plot(sol_nf.t, sol_nf.y[IDX["R"]], color=PAL["indigo"], label="Reserve vesicles")
    ax.plot(sol_nf.t, sol_nf.y[IDX["V"]], color=PAL["teal"], label="Release vesicles")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    ax2.plot(sol_nf.t, sol_nf.y[IDX["Ca"]], color=PAL["pink"], alpha=0.6, label="Ca²⁺ signal")
    ax.set(xlabel="Time (min)", ylabel="Vesicle pool",
           title="(e) Exocytosis Compartments")
    ax2.set_ylabel("Ca²⁺ signal", color=PAL["pink"])
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=7)

    # (f) Gene expression
    ax = axes[1,2]
    ax.plot(sol_nf.t, sol_nf.y[IDX["mRNA"]], color=PAL["orange"], label="mRNA", lw=2)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    ax2.plot(sol_nf.t, sol_nf.y[IDX["Protein"]], color=PAL["red"], label="Protein", ls="--", lw=2)
    ax.set(xlabel="Time (min)", ylabel="mRNA level",
           title="(f) IVTT Gene Expression")
    ax2.set_ylabel("Protein level", color=PAL["red"])
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)

    plt.tight_layout()
    if save: plt.savefig("fig4_combined_model.png")
    plt.close()

    return sol_nf, sol_f, geo


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5 — ECONOMIC & SCALE-UP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def figure5(sol, geo, t_end=120.0, save=True):
    econ = economic_analysis(sol, geo, t_end)

    # Scale-up sweep: vary n_cells
    cell_counts = np.logspace(9, 12, 10)
    annual_kgs = []
    cost_per_gs = []
    p, exo, gene, sc = default_params(), default_exo(), default_gene(), default_scale()

    for nc in cell_counts:
        # Scale batch volume so cells never exceed 20% of reactor
        r = 10e-6
        Vc = (4/3)*np.pi*r**3
        min_batch_L = nc * Vc * 1e3 / 0.20  # cells occupy ≤20%
        batch_L = max(1.0, min_batch_L)
        g = default_geometry(n_cells=nc, batch_L=batch_L)
        s = simulate(p, g, exo, gene, sc, t_end=t_end, n_pts=200)
        e = economic_analysis(s, g, t_end)
        annual_kgs.append(e["annual_kg"])
        cost_per_gs.append(e["cost_per_g"])

    annual_kgs = np.array(annual_kgs)
    cost_per_gs = np.array(cost_per_gs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Economic & Scale-Up Analysis", fontsize=13, fontweight="bold")

    # (a) Annual production vs cell count
    ax = axes[0]
    ax.plot(cell_counts, annual_kgs, color=PAL["dkgreen"], lw=2.5)
    ax.axhline(100_000, color=PAL["red"], ls=":", lw=1.3, label="Global demand (~100,000 kg/yr)")
    ax.fill_between(cell_counts, annual_kgs, 0, alpha=0.06, color=PAL["dkgreen"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Cells per bioreactor batch", ylabel="Annual production (kg/yr)",
           title="(a) Scale-Up: Annual Heparin Output")
    ax.legend(fontsize=8)

    # (b) Cost per gram vs cell count
    ax = axes[1]
    valid = cost_per_gs < 1e6
    ax.plot(cell_counts[valid], cost_per_gs[valid], color=PAL["blue"], lw=2.5)
    ax.axhline(200, color=PAL["amber"], ls="--", lw=1.3, label="Porcine heparin (~$200/g)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Cells per bioreactor batch", ylabel="Cost per gram ($/g)",
           title="(b) Unit Cost vs. Scale")
    ax.legend(fontsize=8)

    # (c) Summary table as text box
    ax = axes[2]
    ax.axis("off")
    rows = [
        ("Parameter", "Value"),
        ("─"*25, "─"*20),
        ("Batch volume", f"{econ['V_out_L']:.1f} L"),
        ("Cells per batch", f"{econ['n_cells']:.1e}"),
        ("Batch duration", f"{econ['batch_hr']:.1f} hr"),
        ("Secreted heparin", f"{econ['S_final_mM']*1e3:.2f} µM"),
        ("Heparin per batch", f"{econ['mass_mg_batch']:.2f} mg"),
        ("Batches per year", f"{econ['batches_per_yr']:.0f}"),
        ("Annual production", f"{econ['annual_kg']:.4f} kg"),
        ("% global demand", f"{econ['pct_global_demand']:.4f}%"),
        ("─"*25, "─"*20),
        ("Cost per batch", f"${econ['cost_per_batch']:.2f}"),
        ("Synthetic cost/g", f"${econ['cost_per_g']:.2f}"),
        ("Porcine cost/g", f"~${econ['porcine_cost_g']:.0f}"),
    ]
    text = "\n".join(f"  {r[0]:.<28s} {r[1]}" for r in rows)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    ax.set_title("(c) Baseline Economics Summary", fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save: plt.savefig("fig5_economics.png")
    plt.close()

    return econ


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6 — OPTIMIZATION: PAPS & GLUCOSE SWEEP HEATMAP
# ═══════════════════════════════════════════════════════════════════════
def figure6(save=True):
    paps_range = np.linspace(5, 60, 8)      # initial PAPS (mM)
    gluc_range = np.linspace(1, 20, 8)      # initial external glucose (mM)
    hep_out = np.zeros((len(paps_range), len(gluc_range)))

    p, exo, gene, sc = default_params(), default_exo(), default_gene(), default_scale()
    t_end = 60.0

    for i, paps0 in enumerate(paps_range):
        for j, g0 in enumerate(gluc_range):
            geo = default_geometry()
            y0 = default_y0(geo)
            y0[6]  = paps0        # PAPS
            y0[8]  = 0.5          # internal glucose stays default
            y0[39] = g0           # G_out
            sol = simulate(p, geo, exo, gene, sc, t_end=t_end, y0=y0, n_pts=200)
            hep_out[i,j] = sol.y[IDX["S"], -1] * 1e3  # µM secreted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Bioreactor Optimization: Parameter Sweeps",
                 fontsize=13, fontweight="bold")

    # Heatmap
    im = ax1.imshow(hep_out, origin="lower", aspect="auto",
                    extent=[gluc_range[0], gluc_range[-1], paps_range[0], paps_range[-1]],
                    cmap="YlGn")
    cb = plt.colorbar(im, ax=ax1, shrink=0.85)
    cb.set_label("Secreted heparin (µM)", fontsize=9)
    ax1.set(xlabel="Initial extracellular glucose (mM)",
            ylabel="Initial PAPS (mM)",
            title="(a) Secreted Heparin vs. PAPS & Glucose")

    # Marginal: fix glucose=5 mM, sweep PAPS
    mid_g = len(gluc_range)//2
    ax2.plot(paps_range, hep_out[:, mid_g], color=PAL["dkgreen"], lw=2.5,
             label=f"Glucose = {gluc_range[mid_g]:.0f} mM")
    # Fix PAPS=20 mM, sweep glucose
    mid_p = np.argmin(np.abs(paps_range - 20))
    ax2_r = ax2.twinx()
    ax2_r.spines["right"].set_visible(True); ax2_r.spines["top"].set_visible(False)
    ax2_r.plot(gluc_range, hep_out[mid_p, :], color=PAL["blue"], lw=2.5, ls="--",
               label=f"PAPS = {paps_range[mid_p]:.0f} mM")
    ax2.set(xlabel="Sweep variable (mM)", ylabel="Secreted heparin (µM) — PAPS sweep",
            title="(b) Marginal Effects")
    ax2_r.set_ylabel("Secreted heparin (µM) — Glucose sweep", color=PAL["blue"])
    lines = ax2.get_lines() + ax2_r.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8)

    plt.tight_layout()
    if save: plt.savefig("fig6_optimization.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  HEPARIN SYNTHETIC-CELL BIOREACTOR — CONSOLIDATED MODEL")
    print("="*60)

    print("\n▸ Figure 1: System overview (simplified model)...")
    figure1()

    print("▸ Figure 2: Failure modes & DNA dose-response...")
    figure2()

    print("▸ Figure 3: Monte Carlo sensitivity analysis...")
    figure3()

    print("▸ Figure 4: Full combined model (batch vs fed-batch)...")
    sol_nf, sol_f, geo = figure4()

    print("\n▸ Figure 5: Economic & scale-up analysis...")
    econ = figure5(sol_nf, geo, t_end=120.0)

    print("\n▸ Figure 6: Optimization parameter sweeps...")
    figure6()

    # Print final summary
    print(f"\n{'='*60}")
    print(f"  PRODUCTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Secreted heparin (baseline):  {econ['S_final_mM']*1e3:.4f} µM")
    print(f"  Mass per batch:               {econ['mass_mg_batch']:.4f} mg")
    print(f"  Annual production (1 reactor): {econ['annual_kg']*1e3:.4f} g/yr")
    print(f"  Cost per gram:                ${econ['cost_per_g']:.2f}")
    print(f"  Porcine benchmark:            ~$200/g")
    print(f"\n  All 6 figures saved as PNG files.")
    print(f"{'='*60}")
