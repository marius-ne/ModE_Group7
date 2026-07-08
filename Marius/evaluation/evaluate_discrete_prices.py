"""
Evaluate LP OPEX values on discrete gas and electricity price samples.

The input prices are read from Erdem/results/Sampling/training/lhs_40_samples.csv.
Those prices are stored as EUR/MWh and are converted to EUR/kWh before solving.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.optimization.core import (
    solve_lp_approximated,
    solve_lp_lower,
    solve_lp_upper,
    solve_milp,
)


PRICE_SAMPLES_CSV = ROOT / "Erdem" / "results" / "Sampling" / "training" / "lhs_40_samples.csv"
OUTPUT_CSV = ROOT / "Marius" / "results" / "opex_discrete_prices_lhs_40.csv"
DEMAND_CSV = ROOT / "Erdem" / "energy_demands.csv"


def load_demands():
    demand_df = pd.read_csv(DEMAND_CSV)
    q_d = demand_df["hourly heat demand [kW]"].to_numpy()
    p_d = demand_df["hourly electricity demand [kW]"].to_numpy()
    return q_d, p_d


def main():
    samples = pd.read_csv(PRICE_SAMPLES_CSV)
    q_d, p_d = load_demands()
    strict_demand_satisfaction = True

    rows = []
    for i, sample in samples.iterrows():
        c_G = float(sample["gas_price"]) / 1000
        c_el = float(sample["electricity_price"]) / 1000

        print(
            f"[{i + 1}/{len(samples)}] "
            f"c_G={c_G:.6f} EUR/kWh  c_el={c_el:.6f} EUR/kWh"
        )

        opex_milp, _ = solve_milp(
            q_d,
            p_d,
            c_G,
            c_el,
            mip_gap=1e-2,
            strict_demand_satisfaction=strict_demand_satisfaction,
        )
        opex_lower, _ = solve_lp_lower(
            q_d,
            p_d,
            c_G,
            c_el,
            strict_demand_satisfaction=strict_demand_satisfaction,
        )
        opex_upper, _ = solve_lp_upper(
            q_d,
            p_d,
            c_G,
            c_el,
            strict_demand_satisfaction=strict_demand_satisfaction,
        )
        opex_approx, _ = solve_lp_approximated(
            q_d,
            p_d,
            c_G,
            c_el,
            mode="mean_efficiency",
            strict_demand_satisfaction=strict_demand_satisfaction,
        )

        rows.append({
            "c_G": c_G,
            "c_el": c_el,
            "opex_milp": opex_milp,
            "opex_lp_lower": opex_lower,
            "opex_lp_upper": opex_upper,
            "opex_lp_approx": opex_approx,
        })

        print(
            f"  MILP={opex_milp:,.2f}  LP_lower={opex_lower:,.2f}  "
            f"LP_upper={opex_upper:,.2f}  LP_approx={opex_approx:,.2f}"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
