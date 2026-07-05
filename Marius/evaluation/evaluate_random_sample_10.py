"""
Evaluate MILP and LP OPEX values on the random ratio sample.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MARIUS_DIR = ROOT / "Marius"
sys.path.insert(0, str(MARIUS_DIR))

from formulation_MILP import solve as solve_milp
from formulation_LP_lower import solve as solve_lp_lower
from formulation_LP_upper import solve as solve_lp_upper
from formulation_LP_approximated import solve as solve_lp_approximated


RATIOS_CSV = ROOT / "Erdem" / "results" / "Sampling" / "training" / "lhs_40_samples.csv"
OUTPUT_CSV = ROOT / "Marius" / "results" / "opex_LHS_2D_sample_40_2.csv"


def main():
    samples = pd.read_csv(RATIOS_CSV)

    rows = []
    for i, sample in samples.iterrows():
        c_G = float(sample["gas_price"])/1000 # €/Kwh
        c_electricity = float(sample["electricity_price"])/1000 # €/Kwh
        ratio = c_G / c_electricity if c_electricity != 0 else 0
        c_el = 1 
        
        print(f"[{i + 1}/{len(samples)}] ratio={ratio:.6f} c_G={c_G:.6f}  c_el={c_el:.6f}")

        opex_milp, _ = solve_milp(ratio, c_el, mip_gap=1e-3)
        opex_lower, _ = solve_lp_lower(ratio, c_el)
        opex_upper, _ = solve_lp_upper(ratio, c_el)
        opex_approx, _ = solve_lp_approximated(ratio, c_el, mode="mean_efficiency")

        rows.append({
            "ratio": ratio,
            "c_G": c_G,
            "c_el": c_el,
            "opex_milp": opex_milp,
            "opex_lp_lower": opex_lower,
            "opex_lp_upper": opex_upper,
            "opex_lp_approximated": opex_approx,
            "actual_c_electricity": c_electricity,
        })

        print(
            f"  MILP={opex_milp:,.2f}  LP_lower={opex_lower:,.2f}  "
            f"LP_upper={opex_upper:,.2f}  LP_approximated={opex_approx:,.2f}"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
