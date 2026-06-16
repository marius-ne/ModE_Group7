import numpy as np
import sys
from matplotlib import pyplot as plt

sys.path.append("Marius")  # to import formulation_MILP from parent directory
from formulation_MILP import solve

# Baseline scaling factor
c_el = 1

price_ratios = np.logspace(-3, 3, 20)  # from 0.01 to 100
c_G_values = c_el * price_ratios
opex_values = []

for c_G in c_G_values:
    print(f"c_G={c_G:.4f}  c_el={c_el:.4f}  ratio={c_G/c_el:.2f}")

    opex = solve(c_G, c_el, mip_gap=1e-2)[0]
    print(f"  OPEX = {opex:,.2f}\n")
    opex_values.append(opex)

# Make linear and log plots of OPEX vs price ratio
fig, ax = plt.subplots(2, 1, figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(price_ratios, opex_values, marker="o")
plt.xlabel("Price ratio (c_G / c_el)")
plt.ylabel("OPEX")
plt.title("OPEX vs Price Ratio for MILP Formulation")
plt.grid(True, which="both", ls="--")

plt.subplot(2, 1, 2)
plt.plot(price_ratios, opex_values, marker="o")
plt.xscale("log")
plt.xlabel("Price ratio (c_G / c_el)")
plt.ylabel("OPEX")
plt.title("OPEX vs Price Ratio for MILP Formulation")
plt.grid(True, which="both", ls="--")   
plt.tight_layout()
plt.show()

