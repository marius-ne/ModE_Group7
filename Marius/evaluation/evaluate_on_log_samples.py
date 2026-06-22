import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

sys.path.append("Erdem")
from src.optimization.core import solve_milp, solve_lp_lower, solve_lp_upper, solve_lp_approximated

_demand_df = pd.read_csv(Path("energy_demands.csv"))
_Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
_P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()

c_el = 1.0
strict_demand_satisfaction = True

ratios = pd.read_csv("Erdem/results/Sampling/log_training_samples.csv")["ratios"].values

rows = []
for i, ratio in enumerate(ratios):
    c_G = c_el * ratio
    print(f"[{i+1}/{len(ratios)}] ratio={ratio:.6f}  c_G={c_G:.6f}")

    opex_milp   = solve_milp(_Q_D, _P_D, c_G, c_el, mip_gap=1e-2, strict_demand_satisfaction=strict_demand_satisfaction)[0]
    opex_lower  = solve_lp_lower(_Q_D, _P_D, c_G, c_el, strict_demand_satisfaction=strict_demand_satisfaction)[0]
    opex_upper  = solve_lp_upper(_Q_D, _P_D, c_G, c_el, strict_demand_satisfaction=strict_demand_satisfaction)[0]
    opex_approx = solve_lp_approximated(_Q_D, _P_D, c_G, c_el, mode="mean_efficiency", strict_demand_satisfaction=strict_demand_satisfaction)[0]

    rows.append({
        "ratio":         ratio,
        "opex_milp":     opex_milp,
        "opex_lp_lower": opex_lower,
        "opex_lp_upper": opex_upper,
        "opex_lp_approx":opex_approx,
    })
    print(f"  MILP={opex_milp:,.2f}  LP_lower={opex_lower:,.2f}  LP_upper={opex_upper:,.2f}  LP_approx={opex_approx:,.2f}")

df = pd.DataFrame(rows)
out_path = "Marius/results/evaluation_log_samples.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows to {out_path}")

# --- Sanity-check plot ---
C_MILP      = "#2166AC"
C_LP_LOWER  = "#4DAC26"
C_LP_UPPER  = "#D6604D"
C_LP_APPROX = "#35978F"
MS = 2.5
LEGEND_KW = dict(framealpha=0.85, edgecolor="gray", fontsize=9)
OPEX_YLABEL = r"OPEX $\left[\dfrac{€_{\mathrm{OPEX}}}{€\,/\,\mathrm{kWh}}\right]$"

ratios_arr      = np.asarray(df["ratio"],         dtype=float)
opex_milp_arr   = np.asarray(df["opex_milp"],     dtype=float)
opex_lower_arr  = np.asarray(df["opex_lp_lower"], dtype=float)
opex_upper_arr  = np.asarray(df["opex_lp_upper"], dtype=float)
opex_approx_arr = np.asarray(df["opex_lp_approx"],dtype=float)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ratios_arr, opex_milp_arr,   color=C_MILP,      linewidth=1.8, linestyle="-",           marker="o", markersize=MS, label="MILP")
ax.plot(ratios_arr, opex_lower_arr,  color=C_LP_LOWER,  linewidth=1.5, linestyle=(0, (4, 2)),   marker="s", markersize=MS, label="LP Lower")
ax.plot(ratios_arr, opex_upper_arr,  color=C_LP_UPPER,  linewidth=1.5, linestyle="-",           marker="^", markersize=MS, label="LP Upper (min of both)")
ax.plot(ratios_arr, opex_approx_arr, color=C_LP_APPROX, linewidth=1.5, linestyle=(0,(3,1,1,1)), marker="*", markersize=MS+1, label="LP Approx (mean eff.)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Price ratio $c_G\,/\,c_{\mathrm{el}}$ $[-]$", fontsize=12)
ax.set_ylabel(OPEX_YLABEL, fontsize=12)
ax.set_title("Sanity check — OPEX vs price ratio (log–log)\nlog-sampled training points")
ax.legend(**LEGEND_KW)
ax.grid(True, which="both", ls="--", alpha=0.5)
fig.tight_layout()

plot_path = "Marius/visualization/evaluation_log_samples.png"
fig.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
plt.show()
