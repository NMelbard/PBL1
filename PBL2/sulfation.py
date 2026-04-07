import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Lumped sulfation model:
# Heparosan + PAPS -> Heparin + PAP
#
# Variables:
# y[0] = Hs   = heparosan
# y[1] = Hp   = heparin
# y[2] = PAPS = sulfate donor
#
# Rate law (mass action):
# v = k * Hs * PAPS

def sulfation_ode(t, y, k):
    Hs, Hp, PAPS = y
    v = k * Hs * PAPS

    dHs_dt = -v
    dHp_dt = v
    dPAPS_dt = -v

    return [dHs_dt, dHp_dt, dPAPS_dt]

# Parameters
k = 0.05   # lumped reaction rate constant

# Initial conditions
Hs0 = 10.0    # initial heparosan
Hp0 = 0.0     # initial heparin
PAPS0 = 12.0  # initial PAPS

y0 = [Hs0, Hp0, PAPS0]

# Time span
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve ODE
sol = solve_ivp(
    fun=lambda t, y: sulfation_ode(t, y, k),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval
)

# Plot
plt.figure(figsize=(8,5))
plt.plot(sol.t, sol.y[0], label='Heparosan')
plt.plot(sol.t, sol.y[1], label='Heparin')
plt.plot(sol.t, sol.y[2], label='PAPS')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Lumped Heparosan → Heparin Sulfation Model')
plt.legend()
plt.tight_layout()
plt.show()
