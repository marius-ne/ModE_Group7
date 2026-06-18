import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.compat import Path

sys.path.append("Marius")  # to import formulation_MILP from parent directory
from formulation_MILP import solve
from formulation_LP_lower import solve as solve_lp_lower
from formulation_LP_upper import solve as solve_lp_upper
from formulation_LP_approximated import solve as solve_lp_approx

# Manually rerun tests by setting below to True
REWRITE = False

# Toggle expensive or optional solves
SOLVE_LP_UPPER_ROUNDED = False   # set False to skip LP upper rounded (can be slow/infeasible)

strict_demand_satisfaction = True  # Set to True to enforce demand satisfaction in all solves (for testing)

# Check if saved file exists
if REWRITE or not (Path(__file__).parent / ".." / "results" / "opex_vs_price_ratio.csv").exists():
    # Baseline scaling factor
    c_el = 1

    price_ratios = np.logspace(-1, 1, 30)  # from 0.01 to 100
    c_G_values = c_el * price_ratios
    opex_milp_values = []
    opex_lp_lower_values = []
    opex_lp_upper_bo_values = []
    opex_lp_upper_chp_values = []
    opex_lp_upper_rounded_values = []
    opex_lp_approx_mean_values = []
    sum_dB_values = []
    sum_dCHP_values = []

    for c_G in c_G_values:
        price_ratio = c_G / c_el
        print(f"c_G={c_G:.4f}  c_el={c_el:.4f}  ratio={price_ratio:.2f}")

        opex_milp, dispatch_milp = solve(c_G, c_el, mip_gap=1e-2, strict_demand_satisfaction=strict_demand_satisfaction)
        print(f"  OPEX (MILP) = {opex_milp:,.2f}")
        opex_milp_values.append(opex_milp)
        sum_dB_values.append((dispatch_milp["dB1"] + dispatch_milp["dB2"]).sum())
        sum_dCHP_values.append((dispatch_milp["dCHP1"] + dispatch_milp["dCHP2"]).sum())

        opex_lp_lower = solve_lp_lower(c_G, c_el, strict_demand_satisfaction=strict_demand_satisfaction)[0]
        print(f"  OPEX (LP lower) = {opex_lp_lower:,.2f}")
        opex_lp_lower_values.append(opex_lp_lower)

        opex_bo = solve_lp_upper(c_G, c_el, mode="boilers_on", strict_demand_satisfaction=strict_demand_satisfaction)[0]
        print(f"  OPEX (LP upper boilers_on) = {opex_bo:,.2f}")
        opex_lp_upper_bo_values.append(opex_bo)

        opex_chp = solve_lp_upper(c_G, c_el, mode="chp_on", strict_demand_satisfaction=strict_demand_satisfaction)[0]
        print(f"  OPEX (LP upper chp_on) = {opex_chp:,.2f}")
        opex_lp_upper_chp_values.append(opex_chp)

        if SOLVE_LP_UPPER_ROUNDED:
            opex_rounded = solve_lp_upper(c_G, c_el, mode="rounded", strict_demand_satisfaction=strict_demand_satisfaction)[0]
        else:
            opex_rounded = float("nan")
        print(f"  OPEX (LP upper rounded) = {opex_rounded:,.2f}")
        opex_lp_upper_rounded_values.append(opex_rounded)

        opex_approx_mean = solve_lp_approx(c_G, c_el, mode="mean_efficiency", strict_demand_satisfaction=strict_demand_satisfaction)[0]
        print(f"  OPEX (LP approx mean_eff) = {opex_approx_mean:,.2f}\n")
        opex_lp_approx_mean_values.append(opex_approx_mean)

    # Save data to CSV for further analysis
    df = pd.DataFrame({
        "price_ratio": price_ratios,
        "opex_milp": opex_milp_values,
        "opex_lp_lower": opex_lp_lower_values,
        "opex_lp_upper_bo": opex_lp_upper_bo_values,
        "opex_lp_upper_chp": opex_lp_upper_chp_values,
        "opex_lp_upper_rounded": opex_lp_upper_rounded_values,
        "opex_lp_approx_mean": opex_lp_approx_mean_values,
        "sum_dB": sum_dB_values,
        "sum_dCHP": sum_dCHP_values,
    })
    df.to_csv("Marius/results/opex_vs_price_ratio.csv", index=False)

else:
    df = pd.read_csv(Path(__file__).parent / ".." / "results" / "opex_vs_price_ratio.csv")
    price_ratios = df["price_ratio"].values
    opex_milp_values = df["opex_milp"].values
    opex_lp_lower_values = df["opex_lp_lower"].values
    opex_lp_upper_bo_values = df["opex_lp_upper_bo"].values
    opex_lp_upper_chp_values = df["opex_lp_upper_chp"].values
    opex_lp_upper_rounded_values = df["opex_lp_upper_rounded"].values
    opex_lp_approx_mean_values = df["opex_lp_approx_mean"].values
    sum_dB_values = df["sum_dB"].values
    sum_dCHP_values = df["sum_dCHP"].values

LEGEND_KW = dict(framealpha=0.85, edgecolor="gray", fontsize=9)
OPEX_YLABEL = r"OPEX $\left[\dfrac{€_{\mathrm{OPEX}}}{€\,/\,\mathrm{kWh}}\right]$"
MS = 2.5   # uniform marker size

# Palette (ColorBrewer-inspired, perceptually distinct)
C_MILP        = "#2166AC"   # strong blue
C_LP_LOWER    = "#4DAC26"   # medium green
C_LP_UB_BO    = "#F4A582"   # light salmon  (faint boilers_on)
C_LP_UB_CHP   = "#FDCC8A"   # light amber   (faint chp_on)
C_LP_UB_MIN   = "#D6604D"   # muted red     (best upper)
C_LP_ROUNDED  = "#9970AB"   # muted purple
C_LP_APPROX   = "#35978F"   # teal
C_DB          = "#1B7837"   # dark green    (delta sums subplot)
C_DCHP        = "#E08214"   # warm orange   (delta sums subplot)

price_ratios = np.asarray(price_ratios)
opex_lp_upper_bo_values  = np.asarray(opex_lp_upper_bo_values,  dtype=float)
opex_lp_upper_chp_values = np.asarray(opex_lp_upper_chp_values, dtype=float)
opex_lp_upper_min_values = np.fmin(opex_lp_upper_bo_values, opex_lp_upper_chp_values)


def _plot_lp_upper(axis):
    """Faint individual modes + bold best-of-two line."""
    axis.plot(price_ratios, opex_lp_upper_bo_values,
              color=C_LP_UB_BO,  linewidth=0.9, alpha=0.6, linestyle=(0, (5, 3)),
              label="LP Upper (boilers ON)")
    axis.plot(price_ratios, opex_lp_upper_chp_values,
              color=C_LP_UB_CHP, linewidth=0.9, alpha=0.6, linestyle=(0, (2, 2)),
              label="LP Upper (CHPs ON)")
    axis.plot(price_ratios, opex_lp_upper_min_values,
              color=C_LP_UB_MIN, linewidth=1.5, linestyle="-", marker="^", markersize=MS,
              label="LP Upper (min of both)")


# Make linear and log plots of OPEX vs price ratio
fig, ax = plt.subplots(3, 1, figsize=(9, 11))
ax[2].sharex(ax[1])

# --- Linear scale ---
ax[0].plot(price_ratios, opex_milp_values,           color=C_MILP,      linewidth=1.8, linestyle="-",      marker="o", markersize=MS, label="MILP")
ax[0].plot(price_ratios, opex_lp_lower_values,       color=C_LP_LOWER,  linewidth=1.5, linestyle=(0,(4,2)),marker="s", markersize=MS, label="LP Lower")
_plot_lp_upper(ax[0])
if SOLVE_LP_UPPER_ROUNDED:
    ax[0].plot(price_ratios, opex_lp_upper_rounded_values, color=C_LP_ROUNDED, linewidth=1.5, linestyle=(0,(1,2)), marker="D", markersize=MS, label="LP Upper (rounded)")
ax[0].plot(price_ratios, opex_lp_approx_mean_values, color=C_LP_APPROX, linewidth=1.5, linestyle=(0,(3,1,1,1)), marker="*", markersize=MS+1, label="LP Approx (mean eff.)")
ax[0].set_ylabel(OPEX_YLABEL, fontsize=12)
ax[0].set_title("OPEX vs Price Ratio — linear scale")
ax[0].legend(**LEGEND_KW)
ax[0].grid(True, which="both", ls="--", alpha=0.5)

# --- Log scale ---
ax[1].plot(price_ratios, opex_milp_values,           color=C_MILP,      linewidth=1.8, linestyle="-",      marker="o", markersize=MS, label="MILP")
ax[1].plot(price_ratios, opex_lp_lower_values,       color=C_LP_LOWER,  linewidth=1.5, linestyle=(0,(4,2)),marker="s", markersize=MS, label="LP Lower")
_plot_lp_upper(ax[1])
if SOLVE_LP_UPPER_ROUNDED:
    ax[1].plot(price_ratios, opex_lp_upper_rounded_values, color=C_LP_ROUNDED, linewidth=1.5, linestyle=(0,(1,2)), marker="D", markersize=MS, label="LP Upper (rounded)")
ax[1].plot(price_ratios, opex_lp_approx_mean_values, color=C_LP_APPROX, linewidth=1.5, linestyle=(0,(3,1,1,1)), marker="*", markersize=MS+1, label="LP Approx (mean eff.)")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_ylabel(OPEX_YLABEL, fontsize=12)
ax[1].set_title("OPEX vs Price Ratio — log scale")
ax[1].legend(**LEGEND_KW)
ax[1].grid(True, which="both", ls="--", alpha=0.5)

# Inset: lower ~10 % of the price-ratio x-axis (leftmost decade tenth)
_INS_X_LO = price_ratios[0]
_INS_X_HI = price_ratios[0] * (price_ratios[-1] / price_ratios[0]) ** 0.10  # 10 % of log span
_ins_mask  = price_ratios <= _INS_X_HI
_ins_series_bounds = [
    opex_lp_lower_values,       # lower bound
    opex_lp_upper_min_values,   # upper bound (best of bo/chp)
]
_ins_valid  = np.concatenate([s[_ins_mask] for s in _ins_series_bounds])
_ins_valid  = _ins_valid[np.isfinite(_ins_valid)]
_INS_Y_LO   = _ins_valid.min() * 0.96
_INS_Y_HI   = _ins_valid.max() * 1.04

axins1 = ax[1].inset_axes([0.56, 0.05, 0.41, 0.35])   # lower-right, ~90 % of former size
axins1.plot(price_ratios, opex_milp_values,           color=C_MILP,     linewidth=1.8, linestyle="-",           marker="o", markersize=MS+0.5)
axins1.plot(price_ratios, opex_lp_lower_values,       color=C_LP_LOWER, linewidth=1.5, linestyle=(0,(4,2)),     marker="s", markersize=MS+0.5)
axins1.plot(price_ratios, opex_lp_upper_bo_values,    color=C_LP_UB_BO, linewidth=0.9, linestyle=(0,(5,3)),     alpha=0.6)
axins1.plot(price_ratios, opex_lp_upper_chp_values,   color=C_LP_UB_CHP,linewidth=0.9, linestyle=(0,(2,2)),     alpha=0.6)
axins1.plot(price_ratios, opex_lp_upper_min_values,   color=C_LP_UB_MIN,linewidth=1.5, linestyle="-",           marker="^", markersize=MS+0.5)
axins1.plot(price_ratios, opex_lp_approx_mean_values, color=C_LP_APPROX,linewidth=1.5, linestyle=(0,(3,1,1,1)), marker="*", markersize=MS+1.5)
if SOLVE_LP_UPPER_ROUNDED:
    axins1.plot(price_ratios, opex_lp_upper_rounded_values, color=C_LP_ROUNDED, linewidth=1.5, linestyle=(0,(1,2)), marker="D", markersize=MS+0.5)
axins1.set_xscale("log")
axins1.set_yscale("log")
axins1.set_xlim(_INS_X_LO, _INS_X_HI)
axins1.set_ylim(_INS_Y_LO, _INS_Y_HI)
axins1.tick_params(axis="both", which="both", labelsize=5, pad=1)
axins1.grid(True, which="both", ls="--", alpha=0.5)
axins1.set_title(f"low-$r$ zoom ($r≤{_INS_X_HI:.2f}$)", fontsize=7, pad=2)
ax[1].indicate_inset_zoom(axins1, edgecolor="0.4")

# Inset: r ∈ [0.8, 1.1] — transition region around r=1
_INS2_X_LO, _INS2_X_HI = 0.8, 1.1
_ins2_inner = np.where((price_ratios >= _INS2_X_LO) & (price_ratios <= _INS2_X_HI))[0]
# expand by one index on each side so flanking interpolated line segments are captured
_i_lo = max(0, _ins2_inner[0] - 1)
_i_hi = min(len(price_ratios) - 1, _ins2_inner[-1] + 1)
_ins2_mask = np.zeros(len(price_ratios), dtype=bool)
_ins2_mask[_i_lo:_i_hi + 1] = True
_ins2_valid = np.concatenate([
    opex_milp_values[_ins2_mask],
    opex_lp_lower_values[_ins2_mask],
    opex_lp_upper_bo_values[_ins2_mask],
    opex_lp_upper_chp_values[_ins2_mask],
    opex_lp_upper_min_values[_ins2_mask],
    opex_lp_approx_mean_values[_ins2_mask],
])
_ins2_valid = _ins2_valid[np.isfinite(_ins2_valid)]
_INS2_Y_LO  = _ins2_valid.min() * 0.996
_INS2_Y_HI  = _ins2_valid.max() * 1.004

axins2 = ax[1].inset_axes([0.38, 0.60, 0.22, 0.33])   # upper area, right of legend, up to ~60 % x
axins2.plot(price_ratios, opex_milp_values,           color=C_MILP,     linewidth=1.8, linestyle="-",           marker="o", markersize=MS+0.5)
axins2.plot(price_ratios, opex_lp_lower_values,       color=C_LP_LOWER, linewidth=1.5, linestyle=(0,(4,2)),     marker="s", markersize=MS+0.5)
axins2.plot(price_ratios, opex_lp_upper_bo_values,    color=C_LP_UB_BO, linewidth=0.9, linestyle=(0,(5,3)),     alpha=0.6)
axins2.plot(price_ratios, opex_lp_upper_chp_values,   color=C_LP_UB_CHP,linewidth=0.9, linestyle=(0,(2,2)),     alpha=0.6)
axins2.plot(price_ratios, opex_lp_upper_min_values,   color=C_LP_UB_MIN,linewidth=1.5, linestyle="-",           marker="^", markersize=MS+0.5)
axins2.plot(price_ratios, opex_lp_approx_mean_values, color=C_LP_APPROX,linewidth=1.5, linestyle=(0,(3,1,1,1)), marker="*", markersize=MS+1.5)
if SOLVE_LP_UPPER_ROUNDED:
    axins2.plot(price_ratios, opex_lp_upper_rounded_values, color=C_LP_ROUNDED, linewidth=1.5, linestyle=(0,(1,2)), marker="D", markersize=MS+0.5)
axins2.set_xscale("log")
axins2.set_yscale("log")
axins2.set_xlim(_INS2_X_LO, _INS2_X_HI)
axins2.set_ylim(_INS2_Y_LO, _INS2_Y_HI)
axins2.tick_params(axis="both", which="both", labelsize=5, pad=1)
axins2.grid(True, which="both", ls="--", alpha=0.5)
axins2.set_title(r"zoom $r\in[0.8,\,1.1]$", fontsize=7, pad=2)
ax[1].indicate_inset_zoom(axins2, edgecolor="0.4")

# --- Delta sums ---
ax[2].plot(price_ratios, sum_dB_values,   color=C_DB,   linewidth=1.5, marker="^", markersize=MS, label=r"$\sum \delta_{\mathrm{B}}$")
ax[2].plot(price_ratios, sum_dCHP_values, color=C_DCHP, linewidth=1.5, marker="v", markersize=MS, label=r"$\sum \delta_{\mathrm{CHP}}$")
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
