

# BASE TEMPLATE from Natalia's hw 



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters 
k_R = 20.0        # mRNA production rate (mRNA/min)
gamma_R = 0.49  # mRNA degradation rate (min^-1) [t_1/2 ~ 2 min]
k_P = 2.28        # Translation rate (protein/mRNA/min)
gamma_P = 0.00057   # Protein degradation/dilution rate (min^-1) [t_1/2 ~ 30 min]



# ODEs
def gene_expression_model(y, t, k_R, gamma_R, k_P, gamma_P):
    mRNA, protein = y
    dmRNA_dt = k_R - gamma_R * mRNA
    dprotein_dt = (k_P * mRNA) - (gamma_P * protein)
    return [dmRNA_dt, dprotein_dt]

# Time ranges
t_mRNA = np.linspace(0, 25, 500) 
t_protein = np.linspace(0, 10000, 1000) 

# Solutions/integrated results
sol_m = odeint(gene_expression_model, [0, 0], t_mRNA, args=(k_R, gamma_R, k_P, gamma_P))
sol_p = odeint(gene_expression_model, [0, 0], t_protein, args=(k_R, gamma_R, k_P, gamma_P))

# Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# mRNA Plot
ax1.plot(t_mRNA, sol_m[:, 0], color='tab:blue', lw=2.5)
ax1.set_ylabel('[mRNA]')
ax1.set_title('[mRNA](t)')
ax1.grid(alpha=0.3)


# Protein Plot
ax2.plot(t_protein, sol_p[:, 1], color='tab:red', lw=2.5)
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('[Protein]')
ax2.set_title('[Protein](t)')
ax2.grid(alpha=0.3)


plt.tight_layout()
plt.show()