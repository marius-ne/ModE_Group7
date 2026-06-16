import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.compat import Path

sys.path.append("Marius")  # to import formulation_MILP from parent directory
from formulation_MILP import solve
from formulation_LP_binary import solve as solve_lp_bin

# Manually rerun tests by setting below to True
REWRITE = False

# Check if saved file exists
if REWRITE or not (Path(__file__).parent / ".." / "results" / "opex_vs_price_ratio.csv").exists():
    # Baseline scaling factor
    c_el = 1

    price_ratios = np.logspace(-1, 1, 30)  # from 0.01 to 100
    c_G_values = c_el * price_ratios
    opex_milp_values = []
    opex_lp_bin_values = []
    sum_dB_values = []
    sum_dCHP_values = []

    for c_G in c_G_values:
        print(f"c_G={c_G:.4f}  c_el={c_el:.4f}  ratio={c_G/c_el:.2f}")

        opex_milp, dispatch_milp = solve(c_G, c_el, mip_gap=1e-2)
        print(f"  OPEX = {opex_milp:,.2f}\n")
        opex_milp_values.append(opex_milp)
        sum_dB_values.append((dispatch_milp["dB1"] + dispatch_milp["dB2"]).sum())
        sum_dCHP_values.append((dispatch_milp["dCHP1"] + dispatch_milp["dCHP2"]).sum())

        opex_lp_bin = solve_lp_bin(c_G, c_el)[0]
        print(f"  OPEX (LP) = {opex_lp_bin:,.2f}\n")
        opex_lp_bin_values.append(opex_lp_bin)

    # Save data to CSV for further analysis
    df = pd.DataFrame({
        "price_ratio": price_ratios,
        "opex_milp": opex_milp_values,
        "opex_lp_bin": opex_lp_bin_values,
        "sum_dB": sum_dB_values,
        "sum_dCHP": sum_dCHP_values,
    })
    df.to_csv("Marius/results/opex_vs_price_ratio.csv", index=False)

else:
    df = pd.read_csv(Path(__file__).parent / ".." / "results" / "opex_vs_price_ratio.csv")
    price_ratios = df["price_ratio"].values
    opex_milp_values = df["opex_milp"].values
    opex_lp_bin_values = df["opex_lp_bin"].values
    sum_dB_values = df["sum_dB"].values
    sum_dCHP_values = df["sum_dCHP"].values

LEGEND_KW = dict(framealpha=0.85, edgecolor="gray", fontsize=10)
OPEX_YLABEL = r"OPEX $\left[\dfrac{€_{\mathrm{OPEX}}}{€\,/\,\mathrm{kWh}}\right]$"

# Make linear and log plots of OPEX vs price ratio
fig, ax = plt.subplots(3, 1, figsize=(12, 14))
ax[2].sharex(ax[1])

# --- Linear scale ---
ax[0].plot(price_ratios, opex_milp_values,   color="blue", marker="o", label="MILP")
ax[0].plot(price_ratios, opex_lp_bin_values, color="red",  marker="s", label="LP")
ax[0].set_ylabel(OPEX_YLABEL)
ax[0].set_title("OPEX vs Price Ratio — linear scale")
ax[0].legend(**LEGEND_KW)
ax[0].grid(True, which="both", ls="--")

# --- Log scale ---
ax[1].plot(price_ratios, opex_milp_values,   color="blue", marker="o", label="MILP")
ax[1].plot(price_ratios, opex_lp_bin_values, color="red",  marker="s", label="LP")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_ylabel(OPEX_YLABEL)
ax[1].set_title("OPEX vs Price Ratio — log scale")
ax[1].legend(**LEGEND_KW)
ax[1].grid(True, which="both", ls="--")

# --- Delta sums ---
ax[2].plot(price_ratios, sum_dB_values,   color="green",  marker="^", label=r"$\sum \delta_{\mathrm{B}}$")
ax[2].plot(price_ratios, sum_dCHP_values, color="orange", marker="v", label=r"$\sum \delta_{\mathrm{CHP}}$")
ax[2].set_xscale("log")
ax[2].set_xlabel(r"Price ratio $c_G\,/\,c_{\mathrm{el}}$ $[-]$")
ax[2].set_ylabel(r"$\sum \delta\;[-]$")
ax[2].set_title(r"MILP Commitment: $\sum \delta_{\mathrm{B}}$ and $\sum \delta_{\mathrm{CHP}}$ vs Price Ratio")
ax[2].legend(**LEGEND_KW)
ax[2].grid(True, which="both", ls="--")

plt.tight_layout()

# Save the figure
fig.savefig("Marius/visualization/opex_vs_price_ratio.png", dpi=300)
plt.close()
