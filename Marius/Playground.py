# %% [Setup]
import sys
import pandas as pd
from pathlib import Path

sys.path.append("Erdem")
sys.path.insert(0, "Marius")

from src.optimization.core import solve_lp_upper
from visualization.plot_lp_upper_dispatch import plot_lp_upper_dispatch
from visualization.plot_heuristic_cases import plot_heuristic_cases

_demand_df = pd.read_csv(Path("energy_demands.csv"))
_Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
_P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()

# %% LP Upper – ratio=0.2 vs ratio=5
C_EL = 1.0

for ratio in [0.2, 5.0]:
    c_G = ratio * C_EL
    print(f"\nSolving LP upper  ratio={ratio}  c_G={c_G:.3f}  c_el={C_EL:.3f}")
    opex, dispatch = solve_lp_upper(_Q_D, _P_D, c_G, C_EL, mode="min")
    print(f"  OPEX = {opex:,.2f}")
    out = plot_lp_upper_dispatch(dispatch, c_G=c_G, c_el=C_EL, opex=opex, mode="min")
    print(f"  Saved → {out}")

# %% chp_on heuristic — output capacity ranges (report figure)
#
# Three panels showing the combined output range [min, max] of the committed
# units for each of the five chp_on decision cases (I–V).
# A constant demand reference line in each panel illustrates:
#   - green bar: demand falls within [min, max] → configuration is feasible
#   - red bar:   min_combined > demand           → unit must be switched off
#   - gray stub: unit type not committed in this case
#
# Demand lines are chosen to highlight the key transitions:
#   D_CH = 350 kW  (between CHP_min_Q ≈ 293 and CHP_max_Q = 470)
#     → Case II (2 CHPs): min = 2·293 = 585 > 350  → infeasible, 2nd CHP off
#     → Case III (1 CHP): 293 < 350 < 470           → feasible
#   D_CE = 220 kW  (between CHP_min_P = 190 and CHP_max_P = 380)
#     → Case II (2 CHPs): min = 2·190 = 380 > 220  → infeasible
#     → Case III (1 CHP): 190 < 220 < 380           → feasible
#   D_B  = 350 kW  (within one-boiler range [106, 530])

out = plot_heuristic_cases(D_B=350.0, D_CH=350.0, D_CE=220.0)
print(f"\nHeuristic figure saved → {out}")

# %%
