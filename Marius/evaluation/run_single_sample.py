"""Run all 4 models on a single sample and print results to terminal."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("Erdem")
from src.optimization.core import solve_milp, solve_lp_lower, solve_lp_upper, solve_lp_approximated

_demand_df = pd.read_csv(Path("energy_demands.csv"))
Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()

c_el  = 1.0
ratio = 1.0          # change this to test different price ratios
c_G   = c_el * ratio
strict_demand_satisfaction = True

print(f"c_el={c_el}  c_G={c_G}  ratio={ratio}\n")

opex_milp,   *_ = solve_milp(Q_D, P_D, c_G, c_el, mip_gap=1e-2, strict_demand_satisfaction=strict_demand_satisfaction)
opex_lower,  *_ = solve_lp_lower(Q_D, P_D, c_G, c_el, strict_demand_satisfaction=strict_demand_satisfaction)
opex_upper,  *_ = solve_lp_upper(Q_D, P_D, c_G, c_el, strict_demand_satisfaction=strict_demand_satisfaction)
opex_approx, *_ = solve_lp_approximated(Q_D, P_D, c_G, c_el, mode="mean_efficiency", strict_demand_satisfaction=strict_demand_satisfaction)

print(f"MILP         : {opex_milp:>12,.2f}")
print(f"LP lower     : {opex_lower:>12,.2f}")
print(f"LP upper     : {opex_upper:>12,.2f}")
print(f"LP approx    : {opex_approx:>12,.2f}")
print()
print(f"Gap lower    : {(opex_milp  - opex_lower) / opex_milp * 100:>8.2f} %")
print(f"Gap upper    : {(opex_upper - opex_milp)  / opex_milp * 100:>8.2f} %")
print(f"Gap approx   : {(opex_approx - opex_milp) / opex_milp * 100:>8.2f} %")
