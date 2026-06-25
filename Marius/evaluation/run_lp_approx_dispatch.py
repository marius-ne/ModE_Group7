"""Run LP approximation on a single ratio sample and save a dispatch plot to Marius/visualization."""

import sys
import pandas as pd
from pathlib import Path

sys.path.append("Erdem")
sys.path.append("Marius")

from src.optimization.core import solve_lp_approximated
from formulation_LP_approximated import plot_dispatch_results

_demand_df = pd.read_csv(Path("energy_demands.csv"))
Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()

c_el  = 1.0
ratio = 1.0          # change this to test different price ratios
c_G   = c_el * ratio

print(f"LP approx  c_el={c_el}  c_G={c_G}  ratio={ratio}")
opex, dispatch = solve_lp_approximated(Q_D, P_D, c_G, c_el, mode="mean_efficiency", strict_demand_satisfaction=True)
print(f"OPEX: {opex:,.2f}")

out = f"Marius/visualization/dispatch_lp_approx_ratio{ratio:.3f}.png"
plot_dispatch_results(dispatch, output_path=out, gas_price=c_G, el_price=c_el, opex=opex, fontsize=12)
print(f"Saved to {out}")
