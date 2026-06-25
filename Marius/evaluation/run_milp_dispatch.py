"""Run MILP on a single ratio sample and save a dispatch plot to Marius/visualization."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.append("Erdem")
sys.path.append("Marius")

from src.optimization.core import solve_milp
from formulation_MILP import plot_dispatch_results

_demand_df = pd.read_csv(Path("energy_demands.csv"))
Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()

c_el  = 1.0
ratio = 2          # change this to test different price ratios
c_G   = c_el * ratio

print(f"MILP  c_el={c_el}  c_G={c_G}  ratio={ratio}")
opex, dispatch = solve_milp(Q_D, P_D, c_G, c_el, mip_gap=1e-2, strict_demand_satisfaction=True)
print(f"OPEX: {opex:,.2f}")

out = f"Marius/visualization/dispatch_milp_ratio{ratio:.3f}.png"
plot_dispatch_results(dispatch, output_path=out, gas_price=c_G, el_price=c_el, opex=opex, fontsize=12)
print(f"Saved to {out}")

# --- standalone delta-sum dispatch plot ---
k = dispatch["k"].to_numpy()
dB1    = dispatch["dB1"].to_numpy()
dB2    = dispatch["dB2"].to_numpy()
dCHP1  = dispatch["dCHP1"].to_numpy()
dCHP2  = dispatch["dCHP2"].to_numpy()
din    = dispatch["din_TES"].to_numpy()
dout   = dispatch["dout_TES"].to_numpy()
d_sum  = dB1 + dB2 + dCHP1 + dCHP2

delta_matrix = np.vstack([dB1, dB2, dCHP1, dCHP2, din, dout])
cmap_uc = mcolors.LinearSegmentedColormap.from_list("uc", ["#ffffe5", "#238443"])

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 6), sharex=True,
                                gridspec_kw={"height_ratios": [3, 2]})

im = ax0.imshow(
    delta_matrix, aspect="auto", interpolation="nearest",
    cmap=cmap_uc, vmin=0, vmax=1,
    extent=[k[0] - 0.5, k[-1] + 0.5, -0.5, 5.5], origin="lower",
)
ax0.set_yticks([0, 1, 2, 3, 4, 5])
ax0.set_yticklabels(["Boiler 1", "Boiler 2", "CHP 1", "CHP 2", "TES in", "TES out"])
ax0.set_title("Unit Commitment Schedule δ")
ax0.set_ylabel("Unit")
fig.colorbar(im, ax=ax0, label="δ [0–1]", fraction=0.015, pad=0.01)

ax1.bar(k, d_sum, width=0.9, color="#2166AC", alpha=0.85, label="Σ δ (production units)")
ax1.set_ylim(0, 4.3)
ax1.set_yticks([0, 1, 2, 3, 4])
ax1.set_ylabel("Active units [-]")
ax1.set_xlabel("Time step k [-]")
ax1.set_title("Sum of Production Unit Commitments (dB1+dB2+dCHP1+dCHP2)")
ax1.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
ax1.legend(loc="upper right", fontsize=9)

fig.suptitle(
    f"MILP  —  c_G={c_G:.3f} €/kWh,  c_el={c_el:.3f} €/kWh"
    f"  |  c_G/c_el = {ratio:.3f}  |  OPEX = {opex:,.2f} €",
    fontsize=12,
)
fig.tight_layout()

out_ds = f"Marius/visualization/dispatch_milp_delta_sum_ratio{ratio:.3f}.png"
fig.savefig(out_ds, dpi=150)
plt.close(fig)
print(f"Saved to {out_ds}")
