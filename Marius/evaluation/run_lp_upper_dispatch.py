"""Solve the LP upper bound on a single price ratio and save the dispatch.

solve_and_save() is the reusable part: plot_lp_upper_dispatch.py imports it to solve on demand
rather than requiring this script to be run first.

    python Marius/evaluation/run_lp_upper_dispatch.py [ratio]
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))
sys.path.insert(0, str(ROOT / "Marius" / "visualization"))

from _dispatch_common import ratio_from_argv, save_dispatch
from src.optimization.core import solve_lp_upper

FORMULATION = "lp_upper"
DEFAULT_RATIO = 0.5
C_EL = 1.0
MODE = "min"  # which heuristic schedule δ is fixed to; see solve_lp_upper
STRICT_DEMAND_SATISFACTION = True

_demand_df = pd.read_csv(ROOT / "energy_demands.csv")
Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()


def solve_and_save(ratio: float) -> tuple[pd.DataFrame, dict]:
    """Solve the LP upper bound at this price ratio, save the dispatch, and return it
    (dispatch, meta)."""
    c_g = C_EL * ratio

    print(f"LP upper  c_el={C_EL}  c_G={c_g}  ratio={ratio}  mode={MODE}")
    opex, dispatch = solve_lp_upper(Q_D, P_D, c_g, C_EL, mode=MODE,
                                    strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION)
    print(f"OPEX: {opex:,.2f}")

    meta = {"formulation": FORMULATION, "ratio": ratio, "c_g": c_g, "c_el": C_EL, "opex": opex,
            "mode": MODE, "strict_demand_satisfaction": STRICT_DEMAND_SATISFACTION}
    save_dispatch(dispatch, meta, FORMULATION, ratio)
    return dispatch, meta


def main():
    solve_and_save(ratio_from_argv(DEFAULT_RATIO))


if __name__ == "__main__":
    main()
