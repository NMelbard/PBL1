import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#still missing a lot of stuff but it includes the glucose -> heparosan process, as well as the heparosan ->
#heparin equations that imadh wrote. It also includes most of the cycling equations as well.

def mm_single_substrate(Vmax, S, Km):
    """
    Standard Michaelis-Menten equation
    """
    return Vmax * S / (Km + S + 1e-12)


def mm_two_substrate(Vmax, A, B, Km_A, Km_B):
    """
    Two-substrate Michaelis-Menten form for reactions that depend on
    both substrates.
    """
    denom = Km_A * Km_B + Km_B * A + Km_A * B + A * B + 1e-12
    return Vmax * A * B / denom


def reversible_epimerization_forward(GlcA_chain, k_eq):
    """
    Quasi-equilibrium relationship used for the GlcA <-> IdoA step.
    This returns the IdoA amount implied by the equilibrium constant.
    """
    return (k_eq * GlcA_chain) / (1.0 + k_eq)


def heparin_pathway_ode(t, y, p):
    """
    Tracks glucose being converted into heparosan precursors, then follows heparosan modification 
    into heparin while also keeping track of the main cofactor and recycling pools involved.
    
    State variables:
        Original downstream pathway:
        y[0]  = HNAc
        y[1]  = HNS
        y[2]  = HEpi
        y[3]  = H2S
        y[4]  = H6S
        y[5]  = H3S
        y[6]  = PAPS
        y[7]  = PAP

        Glucose -> heparosan pathway:
        y[8]  = Glucose
        y[9]  = G6P
        y[10] = F6P
        y[11] = G1P
        y[12] = UDP_Glc
        y[13] = UDP_GlcA
        y[14] = GlcN6P
        y[15] = GlcN1P
        y[16] = GlcNAc1P
        y[17] = UDP_GlcNAc

        Cycling stuff:
        y[18] = Glutamine
        y[19] = Glutamate
        y[20] = NH3
        y[21] = AcetylCoA
        y[22] = CoASH
        y[23] = UTP
        y[24] = UDP
        y[25] = UMP
        y[26] = ATP
        y[27] = ADP
        y[28] = AMP
        y[29] = PPi
        y[30] = Pi
        y[31] = APS
        y[32] = Sulfite
        y[33] = NADH
        y[34] = NAD
    """

    (
        HNAc, HNS, HEpi, H2S, H6S, H3S, PAPS, PAP,
        Glucose, G6P, F6P, G1P, UDP_Glc, UDP_GlcA, GlcN6P, GlcN1P, GlcNAc1P, UDP_GlcNAc,
        Glutamine, Glutamate, NH3, AcetylCoA, CoASH, UTP, UDP, UMP, ATP, ADP, AMP, PPi, Pi,
        APS, Sulfite, NADH, NAD
    ) = y

    # Glucose uptake and early sugar conversion
    v_hex = mm_two_substrate(
        p["Vmax_hex"],
        Glucose,
        ATP,
        p["Km_Glucose_hex"],
        p["Km_ATP_hex"]
    )

    # G6P <-> F6P
    v_PGI_f = mm_single_substrate(p["Vmax_PGI_f"], G6P, p["Km_G6P_PGI"])
    v_PGI_r = mm_single_substrate(p["Vmax_PGI_r"], F6P, p["Km_F6P_PGI"])

    # G6P <-> G1P
    v_PGM_f = mm_single_substrate(p["Vmax_PGM_f"], G6P, p["Km_G6P_PGM"])
    v_PGM_r = mm_single_substrate(p["Vmax_PGM_r"], G1P, p["Km_G1P_PGM"])

    # F6P + glutamine -> glucosamine-6-phosphate + glutamate
    v_GlmS = mm_two_substrate(
        p["Vmax_GlmS"],
        F6P,
        Glutamine,
        p["Km_F6P_GlmS"],
        p["Km_Gln_GlmS"]
    )

    # Glucosamine-6-phosphate -> glucosamine-1-phosphate
    v_GlmM = mm_single_substrate(
        p["Vmax_GlmM"],
        GlcN6P,
        p["Km_GlcN6P_GlmM"]
    )

    # Glucosamine-1-phosphate + acetyl-CoA -> GlcNAc-1-phosphate + CoASH
    v_GlmU_acetyl = mm_two_substrate(
        p["Vmax_GlmU_acetyl"],
        GlcN1P,
        AcetylCoA,
        p["Km_GlcN1P_GlmU_acetyl"],
        p["Km_AcetylCoA_GlmU_acetyl"]
    )

    # GlcNAc-1-phosphate + UTP -> UDP-GlcNAc + PPi
    v_GlmU_UTP = mm_two_substrate(
        p["Vmax_GlmU_UTP"],
        GlcNAc1P,
        UTP,
        p["Km_GlcNAc1P_GlmU_UTP"],
        p["Km_UTP_GlmU_UTP"]
    )

    # G1P + UTP -> UDP-Glc + PPi
    v_GalU = mm_two_substrate(
        p["Vmax_GalU"],
        G1P,
        UTP,
        p["Km_G1P_GalU"],
        p["Km_UTP_GalU"]
    )

    # UDP-Glc + 2 NAD+ -> UDP-GlcA + 2 NADH
    v_KfiD = mm_two_substrate(
        p["Vmax_KfiD"],
        UDP_Glc,
        NAD,
        p["Km_UDP_Glc_KfiD"],
        p["Km_NAD_KfiD"]
    )

    # Heparosan formation depends on both UDP-GlcNAc and UDP-GlcA
    v_heparosan_base = mm_single_substrate(
        p["Vmax_kfiA"],
        UDP_GlcNAc,
        p["Km_kfiA"]
    )

    coupling_glca = UDP_GlcA / (p["Km_kfiC"] + UDP_GlcA + 1e-12)
    v_heparosan = v_heparosan_base * coupling_glca

    # PAP slows down the sulfotransferase steps
    pap_inhibition = 1.0 / (1.0 + PAP / (p["Ki_PAP"] + 1e-12))

    # N-sulfation
    vNS = pap_inhibition * mm_two_substrate(
        p["Vmax_NS"],
        HNAc,
        PAPS,
        p["Km_HNAc_NS"],
        p["Km_PAPS_NS"]
    )

    # Epimerization
    vEpi = mm_single_substrate(
        p["Vmax_Epi"],
        HNS,
        p["Km_Epi"]
    )

    # 2-O-sulfation
    inhibitor_2ost = p.get("inhibitor_2OST", 0.0)
    Ki_2OST = p.get("Ki_2OST", np.inf)
    inhibition_2ost = 1.0 / (1.0 + inhibitor_2ost / (Ki_2OST + 1e-12))

    v2OST = pap_inhibition * inhibition_2ost * mm_two_substrate(
        p["Vmax_2OST"],
        HEpi,
        PAPS,
        p["Km_HEpi_2OST"],
        p["Km_PAPS_2OST"]
    )

    # 6-O-sulfation
    v6OST = pap_inhibition * mm_two_substrate(
        p["Vmax_6OST"],
        H2S,
        PAPS,
        p["Km_H2S_6OST"],
        p["Km_PAPS_6OST"]
    )

    # 3-O-sulfation
    inhibitor_3ost = p.get("inhibitor_3OST", 0.0)
    Ki_3OST = p.get("Ki_3OST", np.inf)
    inhibition_3ost = 1.0 / (1.0 + inhibitor_3ost / (Ki_3OST + 1e-12))

    v3OST = pap_inhibition * inhibition_3ost * mm_two_substrate(
        p["Vmax_3OST"],
        H6S,
        PAPS,
        p["Km_H6S_3OST"],
        p["Km_PAPS_3OST"]
    )

    # Glutamate + NH3 + ATP -> glutamine + ADP + Pi
    v_gln_synth = p["k_gln_synth"] * Glutamate * NH3 * ATP

    # Acetate + CoA + ATP -> acetyl-CoA + AMP + PPi
    v_acs = p["k_acs"] * p["Acetate_pool"] * CoASH * ATP

    # ATP + UDP <-> ADP + UTP
    v_ndk_f = p["k_ndk_f"] * ATP * UDP
    v_ndk_r = p["k_ndk_r"] * ADP * UTP

    # UMP + ATP -> UDP + ADP
    v_umpk = p["k_umpk"] * UMP * ATP

    # PPi -> 2 Pi
    v_ppase = p["k_ppase"] * PPi

    # Sulfite + ATP -> APS + PPi
    v_ATPS = mm_two_substrate(
        p["Vmax_ATPS"],
        Sulfite,
        ATP,
        p["Km_Sulfite_ATPS"],
        p["Km_ATP_ATPS"]
    )

    # APS + ATP -> PAPS + ADP
    v_APSK = mm_two_substrate(
        p["Vmax_APSK"],
        APS,
        ATP,
        p["Km_APS_APSK"],
        p["Km_ATP_APSK"]
    )

    # PAP -> AMP + Pi
    v_PAPase = p["k_PAPase"] * PAP

    # Lumped AMP -> ADP recharge
    v_AMP_recharge = p["k_AMP_recharge"] * AMP * ATP

    # Lumped ADP -> ATP regeneration
    v_ATP_regen = p["k_ATP_regen"] * ADP

    # Lumped NADH -> NAD+ regeneration
    v_NAD_regen = p["k_NAD_regen"] * NADH

    dHNAc_dt = v_heparosan - vNS
    dHNS_dt = vNS - vEpi
    dHEpi_dt = vEpi - v2OST
    dH2S_dt = v2OST - v6OST
    dH6S_dt = v6OST - v3OST
    dH3S_dt = v3OST

    dPAPS_dt = -(vNS + v2OST + v6OST + v3OST) + v_APSK
    dPAP_dt = (vNS + v2OST + v6OST + v3OST) - v_PAPase

    dGlucose_dt = -v_hex
    dG6P_dt = v_hex - v_PGI_f + v_PGI_r - v_PGM_f + v_PGM_r
    dF6P_dt = v_PGI_f - v_PGI_r - v_GlmS
    dG1P_dt = v_PGM_f - v_PGM_r - v_GalU
    dUDP_Glc_dt = v_GalU - v_KfiD
    dUDP_GlcA_dt = v_KfiD - v_heparosan
    dGlcN6P_dt = v_GlmS - v_GlmM
    dGlcN1P_dt = v_GlmM - v_GlmU_acetyl
    dGlcNAc1P_dt = v_GlmU_acetyl - v_GlmU_UTP
    dUDP_GlcNAc_dt = v_GlmU_UTP - v_heparosan

    dGlutamine_dt = -v_GlmS + v_gln_synth
    dGlutamate_dt = v_GlmS - v_gln_synth
    dNH3_dt = -v_gln_synth

    dAcetylCoA_dt = -v_GlmU_acetyl + v_acs
    dCoASH_dt = v_GlmU_acetyl - v_acs

    dUTP_dt = -v_GalU - v_GlmU_UTP + v_ndk_f - v_ndk_r
    dUDP_dt = -v_ndk_f + v_ndk_r + v_umpk
    dUMP_dt = -v_umpk

    dATP_dt = (
        -v_hex
        -v_gln_synth
        -v_acs
        -v_ndk_f + v_ndk_r
        -v_umpk
        -v_ATPS
        -v_APSK
        -v_AMP_recharge
        + v_ATP_regen
    )

    dADP_dt = (
        v_hex
        + v_gln_synth
        + v_ndk_f - v_ndk_r
        + v_umpk
        + v_APSK
        + 2.0 * v_AMP_recharge
        - v_ATP_regen
    )

    dAMP_dt = (
        v_acs
        + v_PAPase
        - v_AMP_recharge
    )

    dPPi_dt = v_GalU + v_GlmU_UTP + v_acs + v_ATPS - v_ppase
    dPi_dt = 2.0 * v_ppase + v_gln_synth + v_PAPase

    dAPS_dt = v_ATPS - v_APSK
    dSulfite_dt = -v_ATPS

    dNADH_dt = 2.0 * v_KfiD - v_NAD_regen
    dNAD_dt = -2.0 * v_KfiD + v_NAD_regen

    return [
        dHNAc_dt,
        dHNS_dt,
        dHEpi_dt,
        dH2S_dt,
        dH6S_dt,
        dH3S_dt,
        dPAPS_dt,
        dPAP_dt,
        dGlucose_dt,
        dG6P_dt,
        dF6P_dt,
        dG1P_dt,
        dUDP_Glc_dt,
        dUDP_GlcA_dt,
        dGlcN6P_dt,
        dGlcN1P_dt,
        dGlcNAc1P_dt,
        dUDP_GlcNAc_dt,
        dGlutamine_dt,
        dGlutamate_dt,
        dNH3_dt,
        dAcetylCoA_dt,
        dCoASH_dt,
        dUTP_dt,
        dUDP_dt,
        dUMP_dt,
        dATP_dt,
        dADP_dt,
        dAMP_dt,
        dPPi_dt,
        dPi_dt,
        dAPS_dt,
        dSulfite_dt,
        dNADH_dt,
        dNAD_dt,
    ]


# these parameters are all based on random stuff -- asked chat to give "reasonable parameters" but I
# would strongly doubt these are generally accurate

params = {
    "UDP_GlcNAc": 10.0,
    "Vmax_kfiA": 1.0,
    "Km_kfiA": 0.5,

    "Vmax_NS": 1.2,
    "Km_HNAc_NS": 0.8,
    "Km_PAPS_NS": 0.5,

    "Vmax_Epi": 0.9,
    "Km_Epi": 0.6,

    "Vmax_2OST": 0.8,
    "Km_HEpi_2OST": 0.7,
    "Km_PAPS_2OST": 0.5,

    "Vmax_6OST": 0.7,
    "Km_H2S_6OST": 0.7,
    "Km_PAPS_6OST": 0.5,

    "Vmax_3OST": 0.4,
    "Km_H6S_3OST": 0.6,
    "Km_PAPS_3OST": 0.5,

    "inhibitor_2OST": 0.0,
    "Ki_2OST": np.inf,
    "inhibitor_3OST": 0.0,
    "Ki_3OST": np.inf,

    "Ki_PAP": 1.0,
    "Km_kfiC": 0.5,

    "Vmax_hex": 2.0,
    "Km_Glucose_hex": 1.0,
    "Km_ATP_hex": 0.5,

    "Vmax_PGI_f": 2.0,
    "Km_G6P_PGI": 0.5,
    "Vmax_PGI_r": 1.5,
    "Km_F6P_PGI": 0.5,

    "Vmax_PGM_f": 1.5,
    "Km_G6P_PGM": 0.5,
    "Vmax_PGM_r": 1.0,
    "Km_G1P_PGM": 0.5,

    "Vmax_GlmS": 1.2,
    "Km_F6P_GlmS": 0.5,
    "Km_Gln_GlmS": 0.5,

    "Vmax_GlmM": 1.0,
    "Km_GlcN6P_GlmM": 0.5,

    "Vmax_GlmU_acetyl": 1.0,
    "Km_GlcN1P_GlmU_acetyl": 0.5,
    "Km_AcetylCoA_GlmU_acetyl": 0.5,

    "Vmax_GlmU_UTP": 1.0,
    "Km_GlcNAc1P_GlmU_UTP": 0.5,
    "Km_UTP_GlmU_UTP": 0.5,

    "Vmax_GalU": 1.0,
    "Km_G1P_GalU": 0.5,
    "Km_UTP_GalU": 0.5,

    "Vmax_KfiD": 0.8,
    "Km_UDP_Glc_KfiD": 0.5,
    "Km_NAD_KfiD": 0.5,

    "k_gln_synth": 0.01,

    "k_acs": 0.01,
    "Acetate_pool": 10.0,

    "k_ndk_f": 0.01,
    "k_ndk_r": 0.005,
    "k_umpk": 0.01,

    "k_ppase": 0.05,

    "Vmax_ATPS": 0.8,
    "Km_Sulfite_ATPS": 0.5,
    "Km_ATP_ATPS": 0.5,

    "Vmax_APSK": 0.8,
    "Km_APS_APSK": 0.5,
    "Km_ATP_APSK": 0.5,

    "k_PAPase": 0.05,
    "k_AMP_recharge": 0.005,

    "k_ATP_regen": 0.3,
    "k_NAD_regen": 0.2,
}


y0 = [
    0.0,   # HNAc
    0.0,   # HNS
    0.0,   # HEpi
    0.0,   # H2S
    0.0,   # H6S
    0.0,   # H3S
    20.0,  # PAPS
    0.0,   # PAP

    50.0,  # Glucose
    0.0,   # G6P
    0.0,   # F6P
    0.0,   # G1P
    0.0,   # UDP_Glc
    0.0,   # UDP_GlcA
    0.0,   # GlcN6P
    0.0,   # GlcN1P
    0.0,   # GlcNAc1P
    0.0,   # UDP_GlcNAc

    10.0,  # Glutamine
    5.0,   # Glutamate
    10.0,  # NH3
    10.0,  # AcetylCoA
    5.0,   # CoASH
    15.0,  # UTP
    5.0,   # UDP
    5.0,   # UMP
    40.0,  # ATP
    10.0,  # ADP
    2.0,   # AMP
    0.0,   # PPi
    10.0,  # Pi
    5.0,   # APS
    10.0,  # Sulfite
    5.0,   # NADH
    20.0   # NAD
]

t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 500)

sol = solve_ivp(
    fun=lambda t, y: heparin_pathway_ode(t, y, params),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method="LSODA"
)

labels = [
    "HNAc", "HNS", "HEpi", "H2S", "H6S", "H3S", "PAPS", "PAP",
    "Glucose", "G6P", "F6P", "G1P", "UDP_Glc", "UDP_GlcA", "GlcN6P", "GlcN1P", "GlcNAc1P", "UDP_GlcNAc",
    "Glutamine", "Glutamate", "NH3", "AcetylCoA", "CoASH", "UTP", "UDP", "UMP",
    "ATP", "ADP", "AMP", "PPi", "Pi", "APS", "Sulfite", "NADH", "NAD"
]

plt.figure(figsize=(12, 7))
for i, label in enumerate(labels):
    plt.plot(sol.t, sol.y[i], label=label)

plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Glucose -> Heparosan -> Heparin Biosynthesis Pathway Model")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
plt.tight_layout()
plt.show()

# Smaller plot showing the main pathway species
major_indices = [8, 13, 17, 0, 1, 2, 3, 4, 5, 6, 7]
major_labels = [labels[i] for i in major_indices]

plt.figure(figsize=(10, 6))
for i in major_indices:
    plt.plot(sol.t, sol.y[i], label=labels[i])

plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Main Species: Glucose to Heparin")
plt.legend()
plt.tight_layout()
plt.show()