"""
7 state variables (mM, time in min):
  glc_ext    external glucose
  glc_int    internal glucose / udp-precursor pool  (upstream fast, qssa)
  mrna       enzyme transcript
  e          lumped enzyme: kfia + ndst + osts
  heparosan  heparin backbone
  paps       active sulfate donor  (resource constraint)
  heparin    final product

parameter stuff:
  [1] shin & noireaux (2012) acs synth biol 1:29-41     gene expression rates
  [2] carruthers (1990) physiol rev 70:1135-76            glut1 km ~ 1.5 mM
  [3] xu et al. (2011) science 334:498-501                cell-free heparin, paps loading
  [4] esko & lindahl (2001) j clin invest 108:169-73      hst km(paps) 10-100 uM range
  [5] deangelis (2007) semin thromb hemost 33:442-8       kfia km range
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

PALETTE = {
    "blue":   "#3B82F6",
    "sky":    "#93C5FD",
    "orange": "#F97316",
    "green":  "#22C55E",
    "dkgreen":"#15803D",
    "purple": "#A855F7",
    "red":    "#EF4444",
    "brown":  "#92400E",
    "gray":   "#6B7280",
}


# ode system:

def ode(t, y, p):
    glc_ext, glc_int, mrna, e, heparosan, paps, heparin = y

    # keep states non-negative
    glc_ext   = max(glc_ext,   0.0)
    glc_int   = max(glc_int,   0.0)
    mrna      = max(mrna,      0.0)
    e         = max(e,         0.0)
    heparosan = max(heparosan, 0.0)
    paps      = max(paps,      0.0)

    # membrane transport — facilitated diffusion (mm kinetics)
    v_import = p["V_t"] * glc_ext / (p["Km_t"] + glc_ext)

    # gene expression — constitutive transcription, first-order degradation
    v_tx    = p["k_tx"]    * p["dna"]   # dna -> mrna  [1]
    v_dmrna = p["k_dm"]    * mrna        # mrna decay   [1]
    v_tl    = p["k_tl"]    * mrna        # mrna -> e    [1]
    v_de    = p["k_de"]    * e           # protein decay [1]

    # heparosan polymerization — kfia rate-limiting; vmax = kcat * e(t)
    v_poly = p["kcat_a"] * e * glc_int / (p["Km_a"] + glc_int)

    # sulfation — bi-substrate mm; paps is the resource constraint  [4]
    denom = (p["Km_h"] * p["Km_p"]
             + p["Km_p"] * heparosan
             + p["Km_h"] * paps
             + heparosan * paps + 1e-12)
    v_sulf = p["kcat_m"] * e * heparosan * paps / denom

    # paps regeneration — lumped atp sulfurylase + aps kinase  [3]
    v_regen = p["k_regen"] * (p["paps_max"] - paps)

    dglc_ext   = -v_import
    dglc_int   =  v_import  - v_poly
    dmrna      =  v_tx      - v_dmrna
    de         =  v_tl      - v_de
    dheparosan =  v_poly    - v_sulf
    dpaps      = -v_sulf    + v_regen
    dheparin   =  v_sulf

    return [dglc_ext, dglc_int, dmrna, de, dheparosan, dpaps, dheparin]


def run(p, y0, t_span=(0, 180), n=600):
    return solve_ivp(
        lambda t, y: ode(t, y, p),
        t_span, y0,
        t_eval=np.linspace(*t_span, n),
        method="LSODA", rtol=1e-8, atol=1e-12,
    )


# parameters:

params = {
    # membrane transport
    "V_t":   0.10,   # mM/min  max import flux
    "Km_t":  1.5,    # mM      glut1 km for glucose [2]; carruthers 1990

    # gene expression — from shin & noireaux 2012 e. coli cell-free system [1]
    "dna":   5e-6,   # mM = 5 nM  typical plasmid loading in cell-free
    "k_tx":  0.10,   # min^-1     transcription rate  (range 0.04-0.10) [1]
    "k_dm":  0.14,   # min^-1     mrna degradation  (half-life ~5 min)  [1]
    "k_tl":  0.06,   # min^-1     translation rate  (range 0.03-0.08)   [1]
    "k_de":  0.004,  # min^-1     protein degradation (half-life ~3 hr) [1]
    # at ss: mrna_ss ~ 3.6 nM, e_ss ~ 54 nM

    # kfia polymerization — rate-limiting step [5]
    "kcat_a": 200.0,  # min^-1   ~3.3 s^-1, consistent with glycosyltransferases [5]
    "Km_a":   0.50,   # mM       udp-glcnac km; mid-range for kfia-type enzymes [5]

    # lumped modification (ndst + 2ost + 6ost + 3ost)
    "kcat_m": 40.0,  # min^-1   ~0.7 s^-1, estimated for lumped sulfotransferases
    "Km_h":   0.05,  # mM       heparosan substrate km (~50 uM) [4]
    "Km_p":   0.05,  # mM       paps km for sulfotransferases (10-100 uM range) [4]

    # paps pool
    "paps_max": 0.05,   # mM = 50 uM  typical cell-free loading [3]
    "k_regen":  0.010,  # min^-1      lumped regeneration rate from atp + sulfate
}

y0 = [
    10.0,   # glc_ext   mM  standard cell culture glucose
    0.0,    # glc_int   mM  starts empty
    0.0,    # mrna
    0.0,    # e         starts at zero — lag phase before heparin production
    0.0,    # heparosan
    0.05,   # paps      mM  added exogenously [3]
    0.0,    # heparin
]

LABELS  = ["Glc_ext", "Glc_int", "mRNA", "E", "Heparosan", "PAPS", "Heparin"]
T_SPAN  = (0, 180)
TARGET  = 10.0   # uM  functional objective


# baseline simulation:

sol = run(params, y0, T_SPAN)
t, y = sol.t, sol.y
final_uM = y[6, -1] * 1000
print(f"baseline — final heparin: {final_uM:.2f} uM  |  "
      f"target >={TARGET} uM: {'met' if final_uM >= TARGET else 'not met'}")


# figure 1: system overview

# precompute annotation anchors
hep_uM      = y[6] * 1000
hepa_uM     = y[4] * 1000
enzyme_nM   = y[3] * 1e6
mrna_nM     = y[2] * 1e6
# time at which heparin first crosses target
_cross      = np.where(hep_uM >= TARGET)[0]
t_cross     = t[_cross[0]] if len(_cross) else None
# time at which enzyme reaches 50% of its final value (end of lag phase)
_lag        = np.where(enzyme_nM >= 0.5 * enzyme_nM[-1])[0]
t_lag       = t[_lag[0]] if len(_lag) else t[-1] / 2

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.42)
fig.suptitle("Synthetic Cell — Heparin Biosynthesis (Baseline)",
             fontsize=15, fontweight="bold")

# ① glucose import
ax = fig.add_subplot(gs[0, 0])
ax.plot(t, y[0], color=PALETTE["blue"],  label="Glc$_{ext}$ (medium)")
ax.plot(t, y[1], color=PALETTE["sky"],   label="Glc$_{int}$ (cytoplasm)", linestyle="--")
ax.fill_between(t, y[0], y[1], alpha=0.07, color=PALETTE["blue"])
ax.set_xlabel("time (min)")
ax.set_ylabel("concentration (mM)")
ax.set_title("① Glucose Import\nGLUT1-type facilitated diffusion (Michaelis–Menten)",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=9)

# ② gene expression — dual y-axis
ax  = fig.add_subplot(gs[0, 1])
ax2 = ax.twinx()
ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(False)
# shade lag phase (0 → t_lag)
ax.axvspan(0, t_lag, alpha=0.06, color=PALETTE["red"], zorder=0)
ax.text(t_lag * 0.45, mrna_nM[-1] * 0.75, "lag\nphase",
        ha="center", fontsize=8.5, color=PALETTE["red"], alpha=0.85, style="italic")
l1, = ax.plot(t,  mrna_nM, color=PALETTE["orange"], label="mRNA (nM)")
l2, = ax2.plot(t, enzyme_nM, color=PALETTE["red"],  label="enzyme (nM)", linestyle="--")
ax.set_xlabel("time (min)")
ax.set_ylabel("mRNA (nM)",    color=PALETTE["orange"])
ax2.set_ylabel("enzyme (nM)", color=PALETTE["red"])
ax.tick_params(axis="y", colors=PALETTE["orange"])
ax2.tick_params(axis="y", colors=PALETTE["red"])
ax.set_title("② Gene Expression → Enzyme Accumulation\nDNA → mRNA (fast) → biosynthetic enzyme (slow)",
             fontweight="bold", fontsize=10)
ax.legend(handles=[l1, l2], loc="center right", fontsize=9)

# ③ heparin production
ax = fig.add_subplot(gs[1, 0])
ax.plot(t, hepa_uM, color=PALETTE["purple"], label="heparosan (backbone)", alpha=0.85)
ax.plot(t, hep_uM,  color=PALETTE["dkgreen"], label="heparin (final product)", linewidth=2.5)
ax.axhline(TARGET, color=PALETTE["dkgreen"], linestyle=":", linewidth=1.5,
           label=f"therapeutic target ({TARGET:.0f} µM)")
ax.fill_between(t, hep_uM, 0, alpha=0.10, color=PALETTE["dkgreen"])
if t_cross is not None:
    ax.axvline(t_cross, color=PALETTE["dkgreen"], linestyle="--", linewidth=1.1, alpha=0.7)
    ax.annotate(f"target met\nt ≈ {t_cross:.0f} min",
                xy=(t_cross, TARGET), xytext=(t_cross + 10, TARGET * 3.5),
                fontsize=8.5, color=PALETTE["dkgreen"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["dkgreen"], lw=1.2))
ax.set_xlabel("time (min)")
ax.set_ylabel("concentration (µM)")
ax.set_title("③ Heparin Production\nKfiA polymerization + NDST/OST sulfation cascade",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=9)

# ④ paps resource constraint
ax = fig.add_subplot(gs[1, 1])
ax.plot(t, y[5] * 1000, color=PALETTE["brown"], label="PAPS (active)")
ax.fill_between(t, y[5] * 1000, 0, alpha=0.12, color=PALETTE["brown"])
ax.axhline(params["paps_max"] * 1000, color=PALETTE["gray"], linestyle="--",
           linewidth=1.2, label="PAPS$_{max}$ (50 µM)")
ax.set_xlabel("time (min)")
ax.set_ylabel("PAPS (µM)")
ax.set_title("④ PAPS: Sulfate Donor Resource Constraint\nconsumed by OSTs, regenerated from ATP + inorganic sulfate",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=9)

plt.savefig("fig1_baseline.png", bbox_inches="tight")
plt.show()


# figure 2: failure modes + dna dose-response

# left: biologically motivated failure modes
failure_scenarios = {
    "baseline":               ({},                    list(y0), PALETTE["gray"]),
    "no PAPS regeneration":   ({"k_regen": 0.0},      list(y0), PALETTE["red"]),
    "glucose-limited (1 mM)": ({},                    [1.0] + y0[1:], PALETTE["orange"]),
    "no gene expression":     ({"k_tx": 0.0},         list(y0), PALETTE["purple"]),
}

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6.0))
fig.suptitle("Synthetic Cell — Design Analysis", fontsize=14, fontweight="bold")

for label, (ov, ic, col) in failure_scenarios.items():
    p_mod = {**params, **ov}
    s = run(p_mod, ic, T_SPAN)
    h_uM = s.y[6] * 1000
    lw = 2.8 if label == "baseline" else 1.8
    ax_l.plot(s.t, h_uM, color=col, label=label, linewidth=lw)

ax_l.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
             label=f"therapeutic target ({TARGET:.0f} µM)")
ax_l.set_xlabel("time (min)")
ax_l.set_ylabel("heparin (µM)")
ax_l.set_title("Failure Modes\nWhat happens when a key component is removed?",
               fontweight="bold", fontsize=10)
ax_l.legend(fontsize=9)

# right: dna dose-response — how much plasmid do you need?
dna_folds  = np.logspace(-1, 1, 40)   # 0.1× to 10× nominal
dna_finals = []
for fold in dna_folds:
    p_dna = {**params, "dna": params["dna"] * fold}
    s = run(p_dna, y0, T_SPAN)
    dna_finals.append(s.y[6, -1] * 1000)
dna_finals = np.array(dna_finals)

# find fold where target is first met
_met_idx = np.where(dna_finals >= TARGET)[0]
fold_cross = dna_folds[_met_idx[0]] if len(_met_idx) else None

ax_r.plot(dna_folds, dna_finals, color=PALETTE["blue"], linewidth=2.2)
ax_r.fill_between(dna_folds, dna_finals, 0, alpha=0.08, color=PALETTE["blue"])
ax_r.axhline(TARGET, color="black", linestyle=":", linewidth=1.4,
             label=f"therapeutic target ({TARGET:.0f} µM)")
if fold_cross is not None:
    ax_r.axvline(fold_cross, color=PALETTE["dkgreen"], linestyle="--", linewidth=1.4)
    ax_r.annotate(f"target met\nat {fold_cross:.2f}× DNA",
                  xy=(fold_cross, TARGET),
                  xytext=(fold_cross * 1.6, TARGET * 2.5),
                  fontsize=9, color=PALETTE["dkgreen"],
                  arrowprops=dict(arrowstyle="->", color=PALETTE["dkgreen"], lw=1.2))
ax_r.axvline(1.0, color=PALETTE["gray"], linestyle="--", linewidth=1.0, alpha=0.7,
             label="nominal DNA (5 nM)")
ax_r.set_xscale("log")
ax_r.set_xlabel("DNA concentration (fold of nominal 5 nM)")
ax_r.set_ylabel("final heparin at t = 180 min (µM)")
ax_r.set_title("Plasmid Dose–Response\nHow much DNA is needed to hit the target?",
               fontweight="bold", fontsize=10)
ax_r.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig2_perturbation.png", bbox_inches="tight")
plt.show()


# figure 3: monte carlo + sensitivity

np.random.seed(42)
N_MC = 300
CV   = 0.25

mc_keys    = ["V_t", "dna", "kcat_a", "Km_p", "k_regen"]
mc_labels  = ["V_transporter", "DNA_conc", "kcat_kfiA", "Km_PAPS", "k_PAPS_regen"]
mc_samples = {k: [] for k in mc_keys}
mc_outputs = []

for _ in range(N_MC):
    p_mc = dict(params)
    for key in mc_keys:
        s = params[key] * np.random.normal(1.0, CV)
        s = max(s, params[key] * 0.01)
        p_mc[key] = s
        mc_samples[key].append(s)
    sol_mc = run(p_mc, y0, T_SPAN)
    mc_outputs.append(sol_mc.y[6, -1] * 1000)

mc_outputs = np.array(mc_outputs)
pct_met    = np.mean(mc_outputs >= TARGET) * 100

# pearson r for tornado chart
corr = {lab: pearsonr(mc_samples[k], mc_outputs)[0]
        for k, lab in zip(mc_keys, mc_labels)}
sorted_labs = sorted(corr, key=lambda k: corr[k])
sorted_vals = [corr[k] for k in sorted_labs]
bar_cols    = [PALETTE["dkgreen"] if v > 0 else PALETTE["red"] for v in sorted_vals]

fig, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(14, 6.0))
fig.suptitle(f"Synthetic Cell — Robustness Under Parameter Uncertainty  (N = {N_MC})",
             fontsize=14, fontweight="bold")

# histogram — split by objective met/not
met     = mc_outputs[mc_outputs >= TARGET]
not_met = mc_outputs[mc_outputs <  TARGET]
bins    = np.linspace(mc_outputs.min(), mc_outputs.max(), 28)
ax_h.hist(not_met, bins=bins, color=PALETTE["red"],     alpha=0.75, label=f"below {TARGET:.0f} µM target")
ax_h.hist(met,     bins=bins, color=PALETTE["dkgreen"],  alpha=0.75, label=f"meets target")
ax_h.axvline(TARGET,                color="black",        linestyle=":",  linewidth=2.0,
             label=f"therapeutic target ({TARGET:.0f} µM)")
ax_h.axvline(np.median(mc_outputs), color=PALETTE["blue"], linestyle="--", linewidth=1.8,
             label=f"median = {np.median(mc_outputs):.1f} µM")
ax_h.set_xlabel("final heparin at t = 180 min (µM)")
ax_h.set_ylabel("number of simulations")
ax_h.set_title(f"Output Distribution\n{pct_met:.0f}% of parameter sets produce sufficient heparin",
               fontweight="bold", fontsize=10)
ax_h.legend(fontsize=9)

# tornado chart — which parameters drive heparin output the most?
ax_t.barh(sorted_labs, sorted_vals, color=bar_cols, alpha=0.85,
          edgecolor="white", height=0.55)
ax_t.axvline(0, color="black", linewidth=1.2)
for i, (lab, val) in enumerate(zip(sorted_labs, sorted_vals)):
    offset = 0.03 if val >= 0 else -0.03
    ha     = "left" if val >= 0 else "right"
    ax_t.text(val + offset, i, f"{val:+.2f}", va="center", fontsize=9, ha=ha)
ax_t.set_xlabel("Pearson r  (linear correlation with final heparin output)")
ax_t.set_title("Parameter Sensitivity — Tornado Chart\nwhich knobs matter most for heparin yield?",
               fontweight="bold", fontsize=10)
ax_t.set_xlim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("fig3_montecarlo.png", bbox_inches="tight")
plt.show()

print(f"\nmonte carlo summary (N={N_MC}, ±{CV*100:.0f}%):")
print(f"  median  {np.median(mc_outputs):.2f} uM")
print(f"  mean    {np.mean(mc_outputs):.2f} +/- {np.std(mc_outputs):.2f} uM")
print(f"  5-95th  {np.percentile(mc_outputs,5):.2f} - {np.percentile(mc_outputs,95):.2f} uM")
print(f"  met     {pct_met:.0f}%")

# references
# [1] shin & noireaux (2012) acs synth biol 1:29-41     doi:10.1021/sb200016s
# [2] carruthers (1990) physiol rev 70:1135-76           glut1 facilitated diffusion
# [3] xu et al. (2011) science 334:498-501               doi:10.1126/science.1207478
# [4] esko & lindahl (2001) j clin invest 108:169-73     doi:10.1172/JCI13530
# [5] deangelis (2007) semin thromb hemost 33:442-8      doi:10.1055/s-2007-982081
