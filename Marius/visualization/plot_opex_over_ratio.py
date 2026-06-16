import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.compat import Path

sys.path.append("Marius")  # to import formulation_MILP from parent directory
from formulation_MILP import solve
from formulation_LP_binary import solve as solve_lp_bin

# Manually rerun tests by setting below to True
REWRITE = True

# Check if saved file exists
if REWRITE or not (Path(__file__).parent / ".." / "results" / "opex_vs_price_ratio.csv").exists():
    # Baseline scaling factor
    c_el = 1

    price_ratios = np.logspace(-1, 1, 30)  # from 0.01 to 100
    c_G_values = c_el * price_ratios
    opex_milp_values = []
    opex_lp_bin_values = []

    for c_G in c_G_values:
        print(f"c_G={c_G:.4f}  c_el={c_el:.4f}  ratio={c_G/c_el:.2f}")

        opex_milp = solve(c_G, c_el, mip_gap=1e-2)[0]
        print(f"  OPEX = {opex_milp:,.2f}\n")
        opex_milp_values.append(opex_milp)

        opex_lp_bin = solve_lp_bin(c_G, c_el)[0]
        print(f"  OPEX (LP) = {opex_lp_bin:,.2f}\n")
        opex_lp_bin_values.append(opex_lp_bin)

    # Save data to CSV for further analysis
    df = pd.DataFrame({
        "price_ratio": price_ratios,
        "opex_milp": opex_milp_values,
        "opex_lp_bin": opex_lp_bin_values,
    })
    df.to_csv("Marius/results/opex_vs_price_ratio.csv", index=False)

else:
    df = pd.read_csv(Path(__file__).parent / ".." / "results" / "opex_vs_price_ratio.csv")
    price_ratios = df["price_ratio"].values
    opex_milp_values = df["opex_milp"].values
    opex_lp_bin_values = df["opex_lp_bin"].values

# Make linear and log plots of OPEX vs price ratio
fig, ax = plt.subplots(2, 1, figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(price_ratios, opex_milp_values, color="blue", marker="o", label="MILP")
plt.plot(price_ratios, opex_lp_bin_values, color="red", marker="s", label="LP")
plt.xlabel("Price ratio (c_G / c_el)")
plt.ylabel("OPEX")
plt.title("OPEX vs Price Ratio for MILP and LP Formulations (linear scale)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.grid(True, which="both", ls="--")

plt.subplot(2, 1, 2)
plt.plot(price_ratios, opex_milp_values, color="blue", marker="o", label="MILP")
plt.plot(price_ratios, opex_lp_bin_values, color="red", marker="s", label="LP")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Price ratio (c_G / c_el)")
plt.ylabel("OPEX")
plt.title("OPEX vs Price Ratio for MILP and LP Formulations (log scale)")
plt.legend()
plt.grid(True, which="both", ls="--")   
plt.tight_layout()

# Save the figure
fig.savefig("Marius/visualization/opex_vs_price_ratio.png", dpi=300)
plt.close()
