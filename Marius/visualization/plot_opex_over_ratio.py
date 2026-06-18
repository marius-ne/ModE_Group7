import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.compat import Path

sys.path.append("Marius")  # to import formulation_MILP from parent directory
from formulation_MILP import solve
from formulation_LP_lower import solve as solve_lp_lower
from formulation_LP_upper import solve as solve_lp_upper

# Manually rerun tests by setting below to True
REWRITE = True

strict_demand_satisfaction = True  # Set to True to enforce demand satisfaction in all solves (for testing)

# Check if saved file exists
if REWRITE or not (Path(__file__).parent / ".." / "results" / "opex_vs_price_ratio.csv").exists():
    # Baseline scaling factor
    c_el = 1

    price_ratios = np.logspace(-1, 1, 30)  # from 0.01 to 100
    c_G_values = c_el * price_ratios
    opex_milp_values = []
    opex_lp_lower_values = []
    opex_lp_upper_values = []
    sum_dB_values = []
    sum_dCHP_values = []

    for c_G in c_G_values:
        price_ratio = c_G / c_el
        print(f"c_G={c_G:.4f}  c_el={c_el:.4f}  ratio={c_G/c_el:.2f}")

        opex_milp, dispatch_milp = solve(c_G, c_el, mip_gap=1e-2, strict_demand_satisfaction=strict_demand_satisfaction)
        print(f"  OPEX = {opex_milp:,.2f}\n")
        opex_milp_values.append(opex_milp)
        sum_dB_values.append((dispatch_milp["dB1"] + dispatch_milp["dB2"]).sum())
        sum_dCHP_values.append((dispatch_milp["dCHP1"] + dispatch_milp["dCHP2"]).sum())

        opex_lp_lower = solve_lp_lower(c_G, c_el, strict_demand_satisfaction=strict_demand_satisfaction)[0]
        print(f"  OPEX (LP) = {opex_lp_lower:,.2f}\n")
        opex_lp_lower_values.append(opex_lp_lower)

        if price_ratio >= 1:
            mode = "boilers_on"
        else:
            mode = "boilers_on"
        opex_lp_upper = solve_lp_upper(c_G, c_el, mode=mode, strict_demand_satisfaction=strict_demand_satisfaction)[0]
        print(f"  OPEX (LP upper) = {opex_lp_upper:,.2f}\n")
        opex_lp_upper_values.append(opex_lp_upper)

    # Save data to CSV for further analysis
    df = pd.DataFrame({
        "price_ratio": price_ratios,
        "opex_milp": opex_milp_values,
        "opex_lp_lower": opex_lp_lower_values,
        "opex_lp_upper": opex_lp_upper_values,
        "sum_dB": sum_dB_values,
        "sum_dCHP": sum_dCHP_values,
    })
    df.to_csv("Marius/results/opex_vs_price_ratio.csv", index=False)

else:
    df = pd.read_csv(Path(__file__).parent / ".." / "results" / "opex_vs_price_ratio.csv")
    price_ratios = df["price_ratio"].values
    opex_milp_values = df["opex_milp"].values
    opex_lp_lower_values = df["opex_lp_lower"].values
    opex_lp_upper_values = df["opex_lp_upper"].values
    sum_dB_values = df["sum_dB"].values
    sum_dCHP_values = df["sum_dCHP"].values

LEGEND_KW = dict(framealpha=0.85, edgecolor="gray", fontsize=10)
OPEX_YLABEL = r"OPEX $\left[\dfrac{€_{\mathrm{OPEX}}}{€\,/\,\mathrm{kWh}}\right]$"

price_ratios = np.asarray(price_ratios)
opex_lp_upper_values = np.asarray(opex_lp_upper_values)
bo_mask  = price_ratios >= 1
chp_mask = ~bo_mask

# Make linear and log plots of OPEX vs price ratio
fig, ax = plt.subplots(3, 1, figsize=(9, 11))
ax[2].sharex(ax[1])

def _plot_lp_upper(axis):
    """Plot LP Upper line with mode-encoded markers (same color, two marker shapes)."""
    axis.plot(price_ratios, opex_lp_upper_values, color="red", linewidth=1.2, zorder=2)
    axis.plot(price_ratios[chp_mask], opex_lp_upper_values[chp_mask],
              color="red", marker="v", linestyle="none", markersize=6,
              label="LP Upper (CHPs ON)", zorder=3)
    axis.plot(price_ratios[bo_mask], opex_lp_upper_values[bo_mask],
              color="red", marker="^", linestyle="none", markersize=6,
              label="LP Upper (boilers ON)", zorder=3)
    axis.axvline(1.0, color="red", linewidth=0.7, linestyle=":", alpha=0.5)

# --- Linear scale ---
ax[0].plot(price_ratios, opex_milp_values,     color="blue",  marker="o", label="MILP")
ax[0].plot(price_ratios, opex_lp_lower_values, color="green", marker="s", label="LP Lower")
_plot_lp_upper(ax[0])
ax[0].set_ylabel(OPEX_YLABEL, fontsize=12)
ax[0].set_title("OPEX vs Price Ratio — linear scale")
ax[0].legend(**LEGEND_KW)
ax[0].grid(True, which="both", ls="--")

# --- Log scale ---
ax[1].plot(price_ratios, opex_milp_values,     color="blue",  marker="o", label="MILP")
ax[1].plot(price_ratios, opex_lp_lower_values, color="green", marker="s", label="LP Lower")
_plot_lp_upper(ax[1])
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_ylabel(OPEX_YLABEL, fontsize=12)
ax[1].set_title("OPEX vs Price Ratio — log scale")
ax[1].legend(**LEGEND_KW)
ax[1].grid(True, which="both", ls="--")

# --- Delta sums ---
ax[2].plot(price_ratios, sum_dB_values,   color="green",  marker="^", label=r"$\sum \delta_{\mathrm{B}}$")
ax[2].plot(price_ratios, sum_dCHP_values, color="orange", marker="v", label=r"$\sum \delta_{\mathrm{CHP}}$")
ax[2].set_xscale("log")
ax[2].set_xlabel(r"Price ratio $c_G\,/\,c_{\mathrm{el}}$ $[-]$", fontsize=12)
ax[2].set_ylabel(r"$\sum \delta\;[-]$", fontsize=12)
ax[2].set_title(r"MILP Commitment: $\sum \delta_{\mathrm{B}}$ and $\sum \delta_{\mathrm{CHP}}$ vs Price Ratio")
ax[2].legend(**LEGEND_KW)
ax[2].grid(True, which="both", ls="--")

plt.tight_layout()

# Save the figure
fig.savefig("Marius/visualization/opex_vs_price_ratio.png", dpi=300)
plt.close()
