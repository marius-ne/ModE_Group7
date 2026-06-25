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


RATIOS_CSV = ROOT / "Erdem" / "results" / "Sampling" / "random_sample_10.csv"
OUTPUT_CSV = ROOT / "Marius" / "results" / "opex_random_sample_10.csv"


def main():
    ratios = pd.read_csv(RATIOS_CSV)["ratios"].values
    c_el = 1.0

    rows = []
    for i, ratio in enumerate(ratios):
        c_G = float(ratio) * c_el
        print(f"[{i + 1}/{len(ratios)}] ratio={ratio:.6f}")

        opex_milp, _ = solve_milp(c_G, c_el, mip_gap=1e-2)
        opex_lower, _ = solve_lp_lower(c_G, c_el)
        opex_upper, _ = solve_lp_upper(c_G, c_el)
        opex_approx, _ = solve_lp_approximated(c_G, c_el, mode="mean_efficiency")

        rows.append({
            "ratio": ratio,
            "c_G": c_G,
            "c_el": c_el,
            "opex_milp": opex_milp,
            "opex_lp_lower": opex_lower,
            "opex_lp_upper": opex_upper,
            "opex_lp_approximated": opex_approx,
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
