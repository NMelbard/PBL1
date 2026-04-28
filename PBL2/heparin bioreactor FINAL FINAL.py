"""
Synthetic Heparin Bioreactor — Coupled 9-State Fed-Batch Model

Modules (in order):
  1. Transcription & Translation  — mRNA and enzyme build-up
  2. Sulfation                    — Heparosan + PAPS → Heparin (lumped)
  3+4. Full Integrated 9-State ODE — coupled synthetic cell model
  5. Porcine vs Synthetic         — head-to-head comparison
  6. Input Stream Optimisation    — glucose, PAPS, DNA sweeps + 2-D heatmap

Fed-batch strategy:
  Bolus feeds of both glucose and PAPS at fixed intervals.
  Implemented by integrating segment-by-segment between pulse times and
  applying an instantaneous concentration jump at each pulse boundary.
  Default: bolus every 120 min (t = 120, 240, 360, 480, 600 min).

State vector (9 states):
  0  glc_ext     — extracellular glucose   (mM)
  1  glc_int     — intracellular glucose   (mM)
  2  mrna        — mRNA                    (mM)
  3  e           — biosynthetic enzyme     (mM)
  4  heparosan   — heparosan backbone      (mM)
  5  paps        — PAPS sulfate donor      (mM)
  6  heparin_ic  — intracellular heparin   (mM)
  7  vesicles    — secretory vesicles      (relative units)
  8  secreted    — secreted heparin        (mM)  ← primary product

All time units : minutes
All conc. units: mM  (axis labels convert to µM / nM where appropriate)

References
----------
[1] Shin & Noireaux (2012) ACS Synth Biol 1:29-41        gene expression rates
[2] Carruthers (1990) Physiol Rev 70:1135-76              GLUT1 Km ~ 1.5 mM
[3] Xu et al. (2011) Science 334:498-501                  cell-free heparin, PAPS loading
[4] Esko & Lindahl (2001) J Clin Invest 108:169-73        HST Km(PAPS) 10-100 uM
[5] DeAngelis (2007) Semin Thromb Hemost 33:442-8         KfiA Km range
[6] Ototani et al. (1981) Carbohyd Res 88:291-303         porcine yield estimates
[7] Linhardt & Gunay (1999) Semin Thromb Hemost 25:5-16   porcine process review
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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

P = {
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


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — TRANSCRIPTION & TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════

def run_txn_tln(k_R=20.0, gamma_R=0.49, k_P=2.28, gamma_P=0.00057,
                t_max=50, dt=0.1):
    """
    Euler integration of mRNA and protein ODEs (E. coli parameters).
    Returns (time, mRNA, protein) arrays.
    """
    time = np.arange(0, t_max, dt)
    m    = np.zeros_like(time)
    prot = np.zeros_like(time)
    for i in range(1, len(time)):
        m[i]    = m[i-1]    + (k_R - gamma_R * m[i-1]) * dt
        prot[i] = prot[i-1] + (k_P * m[i-1] - gamma_P * prot[i-1]) * dt
    return time, m, prot


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — LUMPED SULFATION  (Heparosan + PAPS → Heparin + PAP)
# ══════════════════════════════════════════════════════════════════════════════

def sulfation_ode(t, y, k):
    Hs, Hp, PAPS = y
    v = k * Hs * PAPS
    return [-v, v, -v]


def run_sulfation(k=0.05, Hs0=10.0, Hp0=0.0, PAPS0=12.0,
                  t_span=(0, 100), n=500):
    return solve_ivp(lambda t, y: sulfation_ode(t, y, k),
                     t_span, [Hs0, Hp0, PAPS0],
                     t_eval=np.linspace(*t_span, n))


# ══════════════════════════════════════════════════════════════════════════════
# MODULES 3 + 4 — COUPLED 9-STATE ODE
# ══════════════════════════════════════════════════════════════════════════════

BASE_PARAMS = {
    # glucose import — GLUT1-type facilitated diffusion [2]
    "V_t":       0.10,    "Km_t":      1.5,
    # gene expression — E. coli cell-free [1]
    "dna":       5e-6,    "k_tx":      0.10,   "k_dm":     0.14,
    "k_tl":      0.06,    "k_de":      0.004,
    # KfiA polymerisation [5]
    "kcat_a":    200.0,   "Km_a":      0.50,
    # NDST/OST sulfation cascade — bi-bi MM [3, 4]
    "kcat_m":    40.0,    "Km_h":      0.05,   "Km_p":     0.05,
    # PAPS regeneration [3]
    "paps_max":  0.05,    "k_regen":   0.010,
    # vesicle packaging — Hill kinetics
    "Kpack":     0.00035, "Vmax_pack": 0.1,    "n_pack":   1,
    # secretion and intracellular degradation
    "ksec":      0.26,    "q":         1e-3,    "k_deg":    0.01,
}

BASE_Y0 = [
    10.0,   # glc_ext   — initial glucose feed
     0.0,   # glc_int
     0.0,   # mrna
     0.0,   # enzyme
     0.0,   # heparosan
     0.05,  # paps      — pre-loaded to paps_max
     0.0,   # heparin_ic
     0.0,   # vesicles
     0.0,   # secreted
]

T_END  = 720
T_SPAN = (0, T_END)
TARGET = 10.0   # µM therapeutic target

# ── Fed-batch schedule ────────────────────────────────────────────────────────
# Bolus every 120 min; amounts chosen to partially replenish each substrate.
BOLUS_TIMES = np.arange(120, T_END, 120)   # [120, 240, 360, 480, 600] min
BOLUS_GLC   = 5.0    # mM glucose added per bolus
BOLUS_PAPS  = 0.02   # mM PAPS added per bolus  (= 20 µM)


def _hill(H, K, Vmax, n):
    return Vmax * H**n / (K**n + H**n)


def coupled_ode(t, y, p):
    """9-state coupled ODE (enzyme-driven bi-bi MM sulfation throughout)."""
    glc_ext, glc_int, mrna, e, heparosan, paps, heparin_ic, vesicles, secreted = y

    glc_ext    = max(glc_ext,    0.0)
    glc_int    = max(glc_int,    0.0)
    mrna       = max(mrna,       0.0)
    e          = max(e,          0.0)
    heparosan  = max(heparosan,  0.0)
    paps       = max(paps,       0.0)
    heparin_ic = max(heparin_ic, 0.0)
    vesicles   = max(vesicles,   0.0)

    v_import = p["V_t"]    * glc_ext / (p["Km_t"] + glc_ext)
    v_tx     = p["k_tx"]   * p["dna"]
    v_dmrna  = p["k_dm"]   * mrna
    v_tl     = p["k_tl"]   * mrna
    v_de     = p["k_de"]   * e
    v_poly   = p["kcat_a"] * e * glc_int / (p["Km_a"] + glc_int)

    denom  = (p["Km_h"] * p["Km_p"]
              + p["Km_p"] * heparosan
              + p["Km_h"] * paps
              + heparosan * paps + 1e-12)
    v_sulf = p["kcat_m"] * e * heparosan * paps / denom

    v_regen = p["k_regen"] * (p["paps_max"] - paps)
    v_pack  = _hill(heparin_ic, p["Kpack"], p["Vmax_pack"], p["n_pack"])
    v_sec   = p["ksec"] * vesicles

    return [
        -v_import,
         v_import  - v_poly,
         v_tx      - v_dmrna,
         v_tl      - v_de,
         v_poly    - v_sulf,
        -v_sulf    + v_regen,
         v_sulf    - v_pack - p["k_deg"] * heparin_ic,
        (1.0 / p["q"]) * v_pack - v_sec,
         p["q"]   * v_sec   - p["k_deg"] * secreted,
    ]


def run_batch(params=None, y0=None, t_span=T_SPAN, n=1200):
    """Standard batch integration (no feeding)."""
    p  = BASE_PARAMS if params is None else params
    ic = BASE_Y0     if y0     is None else y0
    return solve_ivp(
        lambda t, y: coupled_ode(t, y, p),
        t_span, ic,
        t_eval=np.linspace(*t_span, n),
        method="LSODA", rtol=1e-8, atol=1e-12,
    )


def run_fedbatch(params=None, y0=None,
                 bolus_times=BOLUS_TIMES,
                 bolus_glc=BOLUS_GLC,
                 bolus_paps=BOLUS_PAPS,
                 t_end=T_END, n_per_seg=200):
    """
    Fed-batch integration with glucose + PAPS bolus pulses.

    Integrates the ODE segment-by-segment.  At each bolus time the solver
    stops, concentrations are updated instantaneously (glc_ext += bolus_glc,
    paps += bolus_paps), then integration resumes from the new state.
    This is the correct way to represent true bolus additions — no
    approximation needed.

    Parameters
    ----------
    bolus_times : array-like  — times (min) at which boluses are added
    bolus_glc   : float       — mM glucose added per bolus
    bolus_paps  : float       — mM PAPS added per bolus
    n_per_seg   : int         — time-points per 120-min segment (scaled)
    """
    p   = BASE_PARAMS if params is None else params
    ic  = list(BASE_Y0 if y0 is None else y0)
    bpt = np.array(bolus_times, dtype=float)

    # segment boundaries
    breakpoints = np.concatenate([[0.0], bpt, [float(t_end)]])
    breakpoints = np.unique(breakpoints)   # remove duplicates, sort

    t_segs, y_segs = [], []

    for k in range(len(breakpoints) - 1):
        t0, t1 = breakpoints[k], breakpoints[k + 1]
        if t1 <= t0:
            continue
        n = max(int(n_per_seg * (t1 - t0) / 120), 20)
        sol = solve_ivp(
            lambda t, y: coupled_ode(t, y, p),
            (t0, t1), ic,
            t_eval=np.linspace(t0, t1, n),
            method="LSODA", rtol=1e-8, atol=1e-12,
        )
        t_segs.append(sol.t)
        y_segs.append(sol.y)

        # carry state forward and apply bolus (if not the last segment)
        ic = list(sol.y[:, -1])
        if t1 in bpt:
            ic[0] = max(ic[0] + bolus_glc,  0.0)   # glc_ext
            ic[5] = max(ic[5] + bolus_paps, 0.0)   # paps

    t_out = np.concatenate(t_segs)
    y_out = np.concatenate(y_segs, axis=1)
    return t_out, y_out


# ══════════════════════════════════════════════════════════════════════════════
# PORCINE REFERENCE MODEL [6, 7]
# ══════════════════════════════════════════════════════════════════════════════

def porcine_yield(t, yield_max=8.0, k_extract=0.03, k_loss=0.002):
    return yield_max * (1 - np.exp(-k_extract * t)) * np.exp(-k_loss * t)


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("Running all simulations …")

# Module 1
time_txn, mrna_txn, prot_txn = run_txn_tln()

# Module 2
sol_sulf = run_sulfation()

# Batch (no feeding)
sol_batch  = run_batch()
t_batch    = sol_batch.t
y_batch    = sol_batch.y
hep_batch  = y_batch[8] * 1000   # secreted µM

# Fed-batch (bolus glucose + PAPS)
t_fb, y_fb = run_fedbatch()
hep_fb     = y_fb[8] * 1000
hepic_fb   = y_fb[6] * 1000
hepa_fb    = y_fb[4] * 1000
enz_fb     = y_fb[3] * 1e6
mrna_fb    = y_fb[2] * 1e6
paps_fb    = y_fb[5]

_lag  = np.where(enz_fb >= 0.5 * enz_fb[-1])[0]
t_lag = t_fb[_lag[0]] if len(_lag) else t_fb[-1] / 2

_cross_fb   = np.where(hep_fb   >= TARGET)[0]
t_cross_fb  = t_fb[_cross_fb[0]]      if len(_cross_fb)   else None
_cross_bat  = np.where(hep_batch >= TARGET)[0]
t_cross_bat = t_batch[_cross_bat[0]]  if len(_cross_bat)  else None

# 2× PAPS fed-batch
t_fb2, y_fb2 = run_fedbatch(params={**BASE_PARAMS, "paps_max": 0.10},
                              bolus_paps=BOLUS_PAPS * 2)
hep_fb2 = y_fb2[8] * 1000

# Porcine reference
t_porc  = np.linspace(0, T_END, 1200)
porc_uM = porcine_yield(t_porc)

print(f"Batch    — final secreted heparin : {hep_batch[-1]:.2f} µM")
print(f"Fed-batch — final secreted heparin : {hep_fb[-1]:.2f} µM")
print(f"Fed-batch 2× PAPS                  : {hep_fb2[-1]:.2f} µM")
print(f"Target {TARGET} µM — "
      f"batch hits at {'t≈'+str(int(t_cross_bat))+' min' if t_cross_bat else 'NEVER'}, "
      f"fed-batch at {'t≈'+str(int(t_cross_fb))+' min' if t_cross_fb else 'NEVER'}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Transcription & Translation  (3 individual windows)
# ══════════════════════════════════════════════════════════════════════════════

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

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Module 1 — mRNA Transcription", fontsize=13, fontweight="bold")
ax.plot(time_txn, mrna_txn, color=P["orange"], label="mRNA")
ax.axhline(m_ss,    color=P["red"],   linestyle="--", label=f"Steady-state ≈ {m_ss:.1f}")
ax.axvline(t_reach, color=P["green"], linestyle="--", label=f"~SS reached t ≈ {t_reach:.1f} min")
ax.set(xlabel="Time (min)", ylabel="mRNA level")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Module 1 — Protein Translation", fontsize=13, fontweight="bold")
ax.plot(time_txn, prot_txn, color=P["blue"], label="Protein")
ax.plot(t_line, P_tangent, linestyle="--", color=P["orange"],
        label=f"Slope ≈ {dP_mid:.1f} proteins/min")
ax.set(xlabel="Time (min)", ylabel="Protein level")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Module 1 — mRNA vs Protein (Normalised)", fontsize=13, fontweight="bold")
ax.plot(time_txn, mrna_txn / np.max(mrna_txn), color=P["orange"], label="mRNA (norm.)")
ax.plot(time_txn, prot_txn / np.max(prot_txn), color=P["blue"],   label="Protein (norm.)")
ax.set(xlabel="Time (min)", ylabel="Normalised level")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("Figure 1 done.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Sulfation Module  (2 individual windows)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Module 2 — Sulfation Species over Time", fontsize=13, fontweight="bold")
ax.plot(sol_sulf.t, sol_sulf.y[0], color=P["blue"],    label="Heparosan")
ax.plot(sol_sulf.t, sol_sulf.y[1], color=P["dkgreen"], label="Heparin")
ax.plot(sol_sulf.t, sol_sulf.y[2], color=P["brown"],   label="PAPS")
ax.set(xlabel="Time (min)", ylabel="Concentration (mM)")
ax.legend()
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Module 2 — Heparosan → Heparin Conversion", fontsize=13, fontweight="bold")
ax.plot(sol_sulf.t, sol_sulf.y[1] / 10.0 * 100, color=P["dkgreen"])
ax.axhline(90, color=P["gray"], linestyle="--", linewidth=1.2, label="90% conversion")
ax.set(xlabel="Time (min)", ylabel="Conversion (%)")
ax.legend()
plt.tight_layout(); plt.show()
print("Figure 2 done.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Fed-batch vs Batch comparison  (key states)
# ══════════════════════════════════════════════════════════════════════════════

def _mark_boluses(ax, ylim_frac=0.06):
    """Draw vertical dashed lines and labels at each bolus time."""
    ylo, yhi = ax.get_ylim()
    for bt in BOLUS_TIMES:
        ax.axvline(bt, color=P["orange"], linestyle=":", linewidth=1.2, alpha=0.7)
    # single legend entry
    ax.axvline(BOLUS_TIMES[0], color=P["orange"], linestyle=":",
               linewidth=1.2, alpha=0.7, label="Bolus feed")

# 3a — Secreted heparin (primary product)
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Fed-Batch vs Batch — Secreted Heparin (primary product)",
             fontsize=13, fontweight="bold")
ax.plot(t_batch, hep_batch, color=P["gray"],    linestyle="--", linewidth=2.0,
        label=f"Batch (no feed) — final: {hep_batch[-1]:.1f} µM")
ax.plot(t_fb,    hep_fb,    color=P["dkgreen"], linewidth=2.5,
        label=f"Fed-batch (Glc + PAPS bolus) — final: {hep_fb[-1]:.1f} µM")
ax.plot(t_fb2,   hep_fb2,   color=P["blue"],    linewidth=2.0, linestyle="-.",
        label=f"Fed-batch 2× PAPS — final: {hep_fb2[-1]:.1f} µM")
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.5,
           label=f"Therapeutic target ({TARGET:.0f} µM)")
_mark_boluses(ax)
if t_cross_bat:
    ax.annotate(f"Batch hits target\nt ≈ {t_cross_bat:.0f} min",
                xy=(t_cross_bat, TARGET), xytext=(t_cross_bat + 30, TARGET + 4),
                fontsize=8, color=P["gray"],
                arrowprops=dict(arrowstyle="->", color=P["gray"], lw=1.0))
if t_cross_fb:
    ax.annotate(f"Fed-batch hits target\nt ≈ {t_cross_fb:.0f} min",
                xy=(t_cross_fb, TARGET), xytext=(t_cross_fb + 30, TARGET + 8),
                fontsize=8, color=P["dkgreen"],
                arrowprops=dict(arrowstyle="->", color=P["dkgreen"], lw=1.0))
ax.set(xlabel="Time (min)", ylabel="Secreted heparin (µM)", xlim=(0, T_END))
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

# 3b — Extracellular glucose (shows bolus spikes)
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Fed-Batch vs Batch — Extracellular Glucose", fontsize=13, fontweight="bold")
ax.plot(t_batch, y_batch[0], color=P["gray"],  linestyle="--", linewidth=1.8,
        label="Batch")
ax.plot(t_fb,    y_fb[0],    color=P["blue"],  linewidth=2.2,
        label="Fed-batch (bolus spikes visible)")
_mark_boluses(ax)
ax.set(xlabel="Time (min)", ylabel="Glc$_{ext}$ (mM)", xlim=(0, T_END))
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

# 3c — PAPS (shows bolus spikes)
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Fed-Batch vs Batch — PAPS (sulfate donor)", fontsize=13, fontweight="bold")
ax.plot(t_batch, y_batch[5] * 1000, color=P["gray"],  linestyle="--", linewidth=1.8,
        label="Batch")
ax.plot(t_fb,    paps_fb    * 1000, color=P["brown"], linewidth=2.2,
        label="Fed-batch (bolus spikes visible)")
ax.axhline(BASE_PARAMS["paps_max"] * 1000, color=P["gray"], linestyle=":",
           linewidth=1.0, label=f"PAPS$_{{max}}$ ({BASE_PARAMS['paps_max']*1000:.0f} µM)")
_mark_boluses(ax)
ax.set(xlabel="Time (min)", ylabel="PAPS (µM)", xlim=(0, T_END))
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("Figure 3 done.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — All 9 fed-batch states as individual windows
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Fed-Batch — ① Extracellular Glucose", fontsize=13, fontweight="bold")
ax.plot(t_fb, y_fb[0], color=P["blue"], label="Glc$_{ext}$ (mM)")
_mark_boluses(ax)
ax.set(xlabel="Time (min)", ylabel="Concentration (mM)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Fed-Batch — ② Intracellular Glucose", fontsize=13, fontweight="bold")
ax.plot(t_fb, y_fb[1], color=P["sky"], label="Glc$_{int}$ (mM)")
ax.fill_between(t_fb, y_fb[1], 0, alpha=0.10, color=P["sky"])
ax.set(xlabel="Time (min)", ylabel="Concentration (mM)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Fed-Batch — ③ mRNA", fontsize=13, fontweight="bold")
ax.plot(t_fb, mrna_fb, color=P["orange"], label="mRNA (nM)")
ax.axvspan(0, t_lag, alpha=0.06, color=P["red"], zorder=0)
ax.text(t_lag * 0.4, mrna_fb.max() * 0.6, "lag phase",
        ha="center", fontsize=8.5, color=P["red"], style="italic")
ax.set(xlabel="Time (min)", ylabel="mRNA (nM)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Fed-Batch — ④ Biosynthetic Enzyme", fontsize=13, fontweight="bold")
ax.plot(t_fb, enz_fb, color=P["red"], label="Enzyme (nM)")
ax.axvspan(0, t_lag, alpha=0.06, color=P["red"], zorder=0)
ax.text(t_lag * 0.4, enz_fb.max() * 0.5, "lag phase",
        ha="center", fontsize=8.5, color=P["red"], style="italic")
ax.set(xlabel="Time (min)", ylabel="Enzyme (nM)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Fed-Batch — ⑤ Heparosan (backbone)", fontsize=13, fontweight="bold")
ax.plot(t_fb, hepa_fb, color=P["purple"], label="Heparosan (µM)")
ax.fill_between(t_fb, hepa_fb, 0, alpha=0.10, color=P["purple"])
_mark_boluses(ax)
ax.set(xlabel="Time (min)", ylabel="Heparosan (µM)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Fed-Batch — ⑥ PAPS (sulfate donor)", fontsize=13, fontweight="bold")
ax.plot(t_fb, paps_fb * 1000, color=P["brown"], label="PAPS (µM)")
ax.fill_between(t_fb, paps_fb * 1000, 0, alpha=0.12, color=P["brown"])
ax.axhline(BASE_PARAMS["paps_max"] * 1000, color=P["gray"], linestyle="--",
           linewidth=1.2, label=f"PAPS$_{{max}}$ ({BASE_PARAMS['paps_max']*1000:.0f} µM)")
_mark_boluses(ax)
ax.set(xlabel="Time (min)", ylabel="PAPS (µM)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Fed-Batch — ⑦ Intracellular Heparin", fontsize=13, fontweight="bold")
ax.plot(t_fb, hepic_fb, color=P["dkgreen"], label="Heparin$_{ic}$ (µM)")
ax.fill_between(t_fb, hepic_fb, 0, alpha=0.10, color=P["dkgreen"])
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
           label=f"Therapeutic target ({TARGET:.0f} µM)")
ax.set(xlabel="Time (min)", ylabel="Heparin$_{ic}$ (µM)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Fed-Batch — ⑧ Secretory Vesicles", fontsize=13, fontweight="bold")
ax.plot(t_fb, y_fb[7], color=P["purple"], label="Vesicles (rel.)")
ax.fill_between(t_fb, y_fb[7], 0, alpha=0.10, color=P["purple"])
ax.set(xlabel="Time (min)", ylabel="Vesicles (relative units)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Fed-Batch — ⑨ Secreted Heparin (product)", fontsize=13, fontweight="bold")
ax.plot(t_fb, hep_fb, color=P["teal"], label="Secreted heparin (µM)", linewidth=2.5)
ax.fill_between(t_fb, hep_fb, 0, alpha=0.12, color=P["teal"])
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
           label=f"Therapeutic target ({TARGET:.0f} µM)")
if t_cross_fb is not None:
    ax.axvline(t_cross_fb, color=P["dkgreen"], linestyle="--", linewidth=1.1, alpha=0.7)
    ax.annotate(f"Target met\nt ≈ {t_cross_fb:.0f} min",
                xy=(t_cross_fb, TARGET), xytext=(t_cross_fb + 40, TARGET + 6),
                fontsize=8.5, color=P["dkgreen"],
                arrowprops=dict(arrowstyle="->", color=P["dkgreen"], lw=1.2))
_mark_boluses(ax)
ax.set(xlabel="Time (min)", ylabel="Secreted heparin (µM)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()
print("Figure 4 done.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Failure Modes & DNA Dose-Response  (fed-batch)
# ══════════════════════════════════════════════════════════════════════════════

failure_scenarios = {
    "Baseline":               ({},               list(BASE_Y0), P["gray"]),
    "No PAPS regeneration":   ({"k_regen": 0.0}, list(BASE_Y0), P["red"]),
    "Glucose-limited (1 mM)": ({},               [1.0] + BASE_Y0[1:], P["orange"]),
    "No gene expression":     ({"k_tx": 0.0},    list(BASE_Y0), P["purple"]),
}

fig, ax_l = plt.subplots(figsize=(9, 6))
fig.suptitle("Fed-Batch — Failure Mode Analysis", fontsize=13, fontweight="bold")

for label, (ov, ic, col) in failure_scenarios.items():
    t_s, y_s = run_fedbatch(params={**BASE_PARAMS, **ov}, y0=ic)
    ax_l.plot(t_s, y_s[8] * 1000, color=col, label=label,
              linewidth=2.8 if label == "Baseline" else 1.8)
ax_l.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
             label=f"Therapeutic target ({TARGET:.0f} µM)")
_mark_boluses(ax_l)
ax_l.set(xlabel="Time (min)", ylabel="Secreted heparin (µM)")
ax_l.legend(fontsize=9)
plt.tight_layout(); plt.show()

dna_folds  = np.logspace(-1, 1, 40)
dna_finals = np.array([run_fedbatch(
                            params={**BASE_PARAMS, "dna": BASE_PARAMS["dna"] * f})[1][8, -1] * 1000
                        for f in dna_folds])
_met       = np.where(dna_finals >= TARGET)[0]
fold_cross = dna_folds[_met[0]] if len(_met) else None

fig, ax_r = plt.subplots(figsize=(9, 6))
fig.suptitle("Fed-Batch — Plasmid Dose–Response", fontsize=13, fontweight="bold")
ax_r.plot(dna_folds, dna_finals, color=P["blue"], linewidth=2.2)
ax_r.fill_between(dna_folds, dna_finals, 0, alpha=0.08, color=P["blue"])
ax_r.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
             label=f"Therapeutic target ({TARGET:.0f} µM)")
if fold_cross is not None:
    ax_r.axvline(fold_cross, color=P["dkgreen"], linestyle="--", linewidth=1.4)
    ax_r.annotate(f"Target met\nat {fold_cross:.2f}× DNA",
                  xy=(fold_cross, TARGET), xytext=(fold_cross * 1.6, TARGET * 2.5),
                  fontsize=9, color=P["dkgreen"],
                  arrowprops=dict(arrowstyle="->", color=P["dkgreen"], lw=1.2))
ax_r.axvline(1.0, color=P["gray"], linestyle="--", linewidth=1.0, alpha=0.7,
             label="Nominal DNA (5 nM)")
ax_r.set_xscale("log")
ax_r.set(xlabel="DNA concentration (fold of nominal 5 nM)",
         ylabel="Final secreted heparin at t = 720 min (µM)")
ax_r.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("Figure 5 done.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Monte Carlo Robustness  (fed-batch)
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(42)
N_MC    = 300
CV      = 0.25
mc_keys = ["V_t", "dna", "kcat_a", "Km_p", "k_regen"]
mc_labs = ["V_transporter", "DNA_conc", "kcat_KfiA", "Km_PAPS", "k_PAPS_regen"]
mc_samp = {k: [] for k in mc_keys}
mc_out  = []

for _ in range(N_MC):
    p_mc = dict(BASE_PARAMS)
    for key in mc_keys:
        s = max(BASE_PARAMS[key] * np.random.normal(1.0, CV),
                BASE_PARAMS[key] * 0.01)
        p_mc[key] = s
        mc_samp[key].append(s)
    _, y_mc = run_fedbatch(params=p_mc)
    mc_out.append(y_mc[8, -1] * 1000)

mc_out  = np.array(mc_out)
pct_met = np.mean(mc_out >= TARGET) * 100

corr        = {lab: pearsonr(mc_samp[k], mc_out)[0]
               for k, lab in zip(mc_keys, mc_labs)}
sorted_labs = sorted(corr, key=lambda k: corr[k])
sorted_vals = [corr[k] for k in sorted_labs]
bar_cols    = [P["dkgreen"] if v > 0 else P["red"] for v in sorted_vals]

fig, ax_h = plt.subplots(figsize=(9, 6))
fig.suptitle(f"Fed-Batch — Output Distribution Under Parameter Uncertainty  "
             f"(N = {N_MC}, CV = {CV*100:.0f}%)",
             fontsize=13, fontweight="bold")
bins = np.linspace(mc_out.min(), mc_out.max(), 28)
ax_h.hist(mc_out[mc_out <  TARGET], bins=bins, color=P["red"],     alpha=0.75,
          label=f"Below {TARGET:.0f} µM target")
ax_h.hist(mc_out[mc_out >= TARGET], bins=bins, color=P["dkgreen"], alpha=0.75,
          label="Meets target")
ax_h.axvline(TARGET,            color="black",   linestyle=":",  linewidth=2.0,
             label=f"Therapeutic target ({TARGET:.0f} µM)")
ax_h.axvline(np.median(mc_out), color=P["blue"], linestyle="--", linewidth=1.8,
             label=f"Median = {np.median(mc_out):.1f} µM")
ax_h.set(xlabel="Final secreted heparin at t = 720 min (µM)",
         ylabel="Number of simulations")
ax_h.legend(fontsize=9)
plt.tight_layout(); plt.show()

fig, ax_t = plt.subplots(figsize=(9, 6))
fig.suptitle(f"Fed-Batch — Sensitivity Tornado Chart  "
             f"(N = {N_MC}, CV = {CV*100:.0f}%)",
             fontsize=13, fontweight="bold")
ax_t.barh(sorted_labs, sorted_vals, color=bar_cols, alpha=0.85,
          edgecolor="white", height=0.55)
ax_t.axvline(0, color="black", linewidth=1.2)
for i, (lab, val) in enumerate(zip(sorted_labs, sorted_vals)):
    offset = 0.03 if val >= 0 else -0.03
    ax_t.text(val + offset, i, f"{val:+.2f}", va="center", fontsize=9,
              ha="left" if val >= 0 else "right")
ax_t.set(xlabel="Pearson r  (linear correlation with final secreted heparin)",
         xlim=(-1.1, 1.1))
plt.tight_layout(); plt.show()
print(f"Figure 6 done.  MC: median {np.median(mc_out):.2f} µM, "
      f"mean {np.mean(mc_out):.2f} ± {np.std(mc_out):.2f} µM, "
      f"{pct_met:.0f}% meet target.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Porcine vs Synthetic Bioreactor Comparison  (fed-batch)
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig 7a — Yield over time ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Porcine vs Synthetic Heparin — Yield over Time (Batch & Fed-Batch)",
             fontsize=13, fontweight="bold")
ax.plot(t_porc, porc_uM, color=P["orange"], linewidth=2.5, linestyle="--",
        label="Porcine (intestinal mucosa extraction) [6,7]")
ax.plot(t_batch, hep_batch, color=P["gray"],    linewidth=1.8, linestyle="--",
        label=f"Synthetic — batch (no feed): {hep_batch[-1]:.1f} µM")
ax.plot(t_fb,    hep_fb,    color=P["dkgreen"], linewidth=2.5,
        label=f"Synthetic — fed-batch (Glc + PAPS bolus): {hep_fb[-1]:.1f} µM")
ax.plot(t_fb2,   hep_fb2,   color=P["blue"],    linewidth=2.0, linestyle="-.",
        label=f"Synthetic — fed-batch 2× PAPS: {hep_fb2[-1]:.1f} µM")
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.5,
           label=f"Therapeutic target ({TARGET:.0f} µM)")
ax.fill_between(t_fb, hep_fb, porc_uM[:len(t_fb)],
                where=(hep_fb > porc_uM[:len(t_fb)]),
                alpha=0.08, color=P["dkgreen"], label="Synthetic advantage region")
for bt in BOLUS_TIMES:
    ax.axvline(bt, color=P["orange"], linestyle=":", linewidth=1.0, alpha=0.5)
ax.axvline(BOLUS_TIMES[0], color=P["orange"], linestyle=":", linewidth=1.0,
           alpha=0.5, label="Bolus feed times")
if t_cross_fb:
    ax.annotate(f"Fed-batch crosses target\nt ≈ {t_cross_fb:.0f} min",
                xy=(t_cross_fb, TARGET), xytext=(t_cross_fb + 40, TARGET + 5),
                fontsize=8.5, color=P["dkgreen"],
                arrowprops=dict(arrowstyle="->", color=P["dkgreen"], lw=1.2))
ax.set(xlabel="Time (min)", ylabel="Heparin (µM)")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

# ── Fig 7b — Final yield comparison bar chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Porcine vs Synthetic Heparin — Final Yield Comparison",
             fontsize=13, fontweight="bold")
categories = ["Porcine\n(peak)", "Batch\nbaseline", "Fed-batch\nbaseline", "Fed-batch\n2× PAPS"]
values     = [np.max(porc_uM), hep_batch[-1], hep_fb[-1], hep_fb2[-1]]
colours_b  = [P["orange"], P["gray"], P["dkgreen"], P["blue"]]
bars = ax.bar(categories, values, color=colours_b, alpha=0.85, edgecolor="white", width=0.55)
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.5,
           label=f"Target ({TARGET:.0f} µM)")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
            f"{val:.1f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set(ylabel="Heparin yield (µM)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

# ── Fig 7c — PAPS loading vs yield ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Porcine vs Synthetic Heparin — PAPS Loading vs Yield (Fed-Batch)",
             fontsize=13, fontweight="bold")
paps_range  = np.linspace(0.01, 0.15, 25)
paps_yields = np.array([run_fedbatch(params={**BASE_PARAMS, "paps_max": pv})[1][8, -1] * 1000
                         for pv in paps_range])
ax.plot(paps_range * 1000, paps_yields, color=P["brown"], linewidth=2.2)
ax.fill_between(paps_range * 1000, paps_yields, 0, alpha=0.10, color=P["brown"])
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.3)
ax.axvline(BASE_PARAMS["paps_max"] * 1000, color=P["gray"],
           linestyle="--", linewidth=1.1, label="Nominal PAPS (50 µM)")
ax.set(xlabel="PAPS$_{max}$ (µM)", ylabel="Final secreted heparin (µM)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

# ── Fig 7d — Estimated product purity ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Porcine vs Synthetic Heparin — Estimated Product Purity",
             fontsize=13, fontweight="bold")
sources = ["Porcine\n(conventional)", "Porcine\n(GMP-grade)", "Synthetic\n(cell-free)"]
purity  = [72, 85, 99]
risk    = [28, 15,  1]
x = np.arange(len(sources))
ax.bar(x, purity, 0.35, color=[P["orange"], P["teal"], P["dkgreen"]],
       alpha=0.85, label="Structural purity (%)", edgecolor="white")
ax.bar(x, risk,   0.35, bottom=purity, color=P["red"],
       alpha=0.55, label="Contamination / impurity risk (%)", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(sources, fontsize=9)
ax.set(ylabel="Percentage (%)", ylim=(0, 115))
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

# ── Fig 7e — Summary comparison table ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle("Porcine vs Synthetic Heparin — Summary Comparison",
             fontsize=13, fontweight="bold")
ax.axis("off")
rows = [
    ["Metric",               "Porcine",       "Batch",             "Fed-batch",         "Fed-batch\n2× PAPS"],
    ["Peak yield (µM)",      f"{np.max(porc_uM):.1f}", f"{hep_batch[-1]:.1f}",
                              f"{hep_fb[-1]:.1f}",     f"{hep_fb2[-1]:.1f}"],
    ["Meets target?",        "No"  if np.max(porc_uM) < TARGET else "Yes",
                              "Yes" if hep_batch[-1]  >= TARGET else "No",
                              "Yes" if hep_fb[-1]     >= TARGET else "No",
                              "Yes" if hep_fb2[-1]    >= TARGET else "No"],
    ["Time to target (min)", "N/A",
                              f"{t_cross_bat:.0f}" if t_cross_bat else "N/A",
                              f"{t_cross_fb:.0f}"  if t_cross_fb  else "N/A",
                              "—"],
    ["Animal-derived?",      "Yes", "No",       "No",        "No"],
    ["Contamination risk",   "High","Very low", "Very low",  "Very low"],
    ["Scalability",          "Limited","High",  "High",      "High"],
]
table = ax.table(cellText=rows[1:], colLabels=rows[0],
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(7.5)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#D1D5DB")
    if r == 0:
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 1: cell.set_facecolor("#FEF3C7")
    elif c == 2: cell.set_facecolor("#E5E7EB")
    elif c == 3: cell.set_facecolor("#D1FAE5")
    elif c == 4: cell.set_facecolor("#DBEAFE")
    else:        cell.set_facecolor("white")
plt.tight_layout(); plt.show()
print("Figure 7 done.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Input Stream Optimisation  (fed-batch)
# ══════════════════════════════════════════════════════════════════════════════

print("Running input stream optimisation sweeps …")

glc_vals   = np.linspace(0.5, 25, 40)
glc_yields = np.array([run_fedbatch(y0=[g] + BASE_Y0[1:])[1][8, -1] * 1000
                        for g in glc_vals])

bolus_glc_vals = np.linspace(0, 15, 40)
bolus_glc_yields = np.array([run_fedbatch(bolus_glc=bg)[1][8, -1] * 1000
                               for bg in bolus_glc_vals])

bolus_paps_vals = np.linspace(0, 0.08, 40)
bolus_paps_yields = np.array([run_fedbatch(bolus_paps=bp)[1][8, -1] * 1000
                                for bp in bolus_paps_vals])

# 2-D: bolus_glc × bolus_paps heatmap
BG_grid  = np.linspace(0, 15, 20)
BP_grid  = np.linspace(0, 0.08, 20)
Z = np.zeros((len(BP_grid), len(BG_grid)))
for j, bg in enumerate(BG_grid):
    for i, bp in enumerate(BP_grid):
        Z[i, j] = run_fedbatch(bolus_glc=bg, bolus_paps=bp)[1][8, -1] * 1000

print("  … sweeps done")

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Input Stream Optimisation — Initial Glucose Concentration",
             fontsize=13, fontweight="bold")
ax.plot(glc_vals, glc_yields, color=P["blue"], linewidth=2.2)
ax.fill_between(glc_vals, glc_yields, 0, alpha=0.10, color=P["blue"])
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
           label=f"Therapeutic target ({TARGET:.0f} µM)")
ax.axvline(BASE_Y0[0], color=P["gray"], linestyle="--", linewidth=1.1,
           label=f"Nominal initial Glc (10 mM)")
best_glc = glc_vals[np.argmax(glc_yields)]
ax.scatter([best_glc], [np.max(glc_yields)], zorder=5, color=P["orange"], s=70,
           label=f"Optimum ≈ {best_glc:.1f} mM")
ax.set(xlabel="Initial glucose feed (mM)",
       ylabel="Final secreted heparin at t = 720 min (µM)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Input Stream Optimisation — Glucose Bolus Size",
             fontsize=13, fontweight="bold")
ax.plot(bolus_glc_vals, bolus_glc_yields, color=P["blue"], linewidth=2.2)
ax.fill_between(bolus_glc_vals, bolus_glc_yields, 0, alpha=0.10, color=P["blue"])
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
           label=f"Therapeutic target ({TARGET:.0f} µM)")
ax.axvline(BOLUS_GLC, color=P["gray"], linestyle="--", linewidth=1.1,
           label=f"Nominal bolus ({BOLUS_GLC:.0f} mM)")
best_bg = bolus_glc_vals[np.argmax(bolus_glc_yields)]
ax.scatter([best_bg], [np.max(bolus_glc_yields)], zorder=5, color=P["orange"], s=70,
           label=f"Optimum ≈ {best_bg:.1f} mM/bolus")
ax.set(xlabel="Glucose bolus size (mM per pulse)",
       ylabel="Final secreted heparin at t = 720 min (µM)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Input Stream Optimisation — PAPS Bolus Size",
             fontsize=13, fontweight="bold")
ax.plot(bolus_paps_vals * 1000, bolus_paps_yields, color=P["brown"], linewidth=2.2)
ax.fill_between(bolus_paps_vals * 1000, bolus_paps_yields, 0, alpha=0.10, color=P["brown"])
ax.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
           label=f"Therapeutic target ({TARGET:.0f} µM)")
ax.axvline(BOLUS_PAPS * 1000, color=P["gray"], linestyle="--", linewidth=1.1,
           label=f"Nominal bolus ({BOLUS_PAPS*1000:.0f} µM)")
best_bp = bolus_paps_vals[np.argmax(bolus_paps_yields)]
ax.scatter([best_bp * 1000], [np.max(bolus_paps_yields)], zorder=5,
           color=P["orange"], s=70, label=f"Optimum ≈ {best_bp*1000:.0f} µM/bolus")
ax.set(xlabel="PAPS bolus size (µM per pulse)",
       ylabel="Final secreted heparin at t = 720 min (µM)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Input Stream Optimisation — Joint Glucose × PAPS Bolus Optimisation",
             fontsize=13, fontweight="bold")
im = ax.contourf(BG_grid, BP_grid * 1000, Z, levels=25, cmap="YlGn")
plt.colorbar(im, ax=ax, label="Final secreted heparin (µM)", pad=0.02)
ax.contour(BG_grid, BP_grid * 1000, Z, levels=[TARGET],
           colors="black", linewidths=1.8, linestyles="--")
ax.scatter([BOLUS_GLC], [BOLUS_PAPS * 1000],
           marker="o", s=90, color=P["gray"], zorder=5, label="Nominal bolus sizes")
bj, bi = np.unravel_index(np.argmax(Z), Z.shape)
ax.scatter([BG_grid[bi]], [BP_grid[bj] * 1000],
           marker="*", s=150, color=P["orange"], zorder=6,
           label=f"Grid optimum ({BG_grid[bi]:.1f} mM Glc, {BP_grid[bj]*1000:.0f} µM PAPS)")
ax.set(xlabel="Glucose bolus size (mM per pulse)",
       ylabel="PAPS bolus size (µM per pulse)")
ax.legend(fontsize=8.5, loc="lower right")
plt.tight_layout(); plt.show()
print("Figure 8 done.")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  SYNTHETIC HEPARIN BIOREACTOR — FINAL SUMMARY  (fed-batch)")
print("=" * 65)
print(f"  Fed-batch strategy    : bolus every 120 min")
print(f"  Glucose bolus         : {BOLUS_GLC:.1f} mM per pulse  "
      f"({len(BOLUS_TIMES)} pulses → +{BOLUS_GLC*len(BOLUS_TIMES):.0f} mM total)")
print(f"  PAPS bolus            : {BOLUS_PAPS*1000:.0f} µM per pulse  "
      f"({len(BOLUS_TIMES)} pulses → +{BOLUS_PAPS*len(BOLUS_TIMES)*1000:.0f} µM total)")
print(f"  Batch final secreted  : {hep_batch[-1]:.2f} µM")
print(f"  Fed-batch final       : {hep_fb[-1]:.2f} µM  "
      f"(+{hep_fb[-1]-hep_batch[-1]:.1f} µM vs batch)")
print(f"  Fed-batch 2× PAPS     : {hep_fb2[-1]:.2f} µM")
print(f"  Porcine peak yield    : {np.max(porc_uM):.2f} µM")
print(f"  Therapeutic target    : {TARGET:.1f} µM")
if t_cross_bat:
    print(f"  Batch hits target     : t ≈ {t_cross_bat:.0f} min")
if t_cross_fb:
    print(f"  Fed-batch hits target : t ≈ {t_cross_fb:.0f} min")
print(f"  Monte Carlo (N={N_MC}): median {np.median(mc_out):.2f}, "
      f"mean {np.mean(mc_out):.2f} ± {np.std(mc_out):.2f} µM, "
      f"{pct_met:.0f}% meet target")
print("=" * 65)
