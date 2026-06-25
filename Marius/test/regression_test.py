"""
Regression test suite for the four Marius dispatch formulations.

Usage
-----
  # (Re-)generate baselines from scratch and save to file:
  python Marius/test/regression_test.py --generate

  # Check current formulations against saved baselines:
  python Marius/test/regression_test.py

The baseline file is stored at Marius/results/regression_baselines.json.
Each entry records OPEX (total cost) for a (c_G, c_el) price pair.

Tolerances
----------
  MILP      : 2e-2 relative  (MIPGap=1e-2)
  LP_lower  : 1e-4 relative  (pure LP, deterministic up to numerical noise)
  LP_upper  : 1e-4 relative  (pure LP, deterministic up to numerical noise)
  LP_approx : 1e-4 relative  (pure LP, deterministic up to numerical noise)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("Erdem")
from src.optimization.core import solve_milp, solve_lp_lower, solve_lp_upper, solve_lp_approximated

_demand_df = pd.read_csv(Path("energy_demands.csv"))
_Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
_P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASELINE_FILE = Path(__file__).parent / ".." / "results" / "regression_baselines.json"

PRICE_PAIRS: list[tuple[float, float]] = [
    (0.08, 0.08),   # both cheap: CHPs run flat-out, minimal grid
    (0.10, 0.25),   # cheap gas, moderate electricity: CHPs preferred
    (0.16, 0.30),   # mid-range: balanced dispatch
    (0.22, 0.12),   # expensive gas, cheap electricity: more grid import
    (0.28, 0.50),   # both expensive: tight optimisation
]

MILP_MIP_GAP: float = 1e-2

FORMULATIONS: dict[str, callable] = {
    "MILP":      lambda c_G, c_el: solve_milp(_Q_D, _P_D, c_G, c_el, mip_gap=MILP_MIP_GAP),
    "LP_lower":  lambda c_G, c_el: solve_lp_lower(_Q_D, _P_D, c_G, c_el),
    "LP_upper":  lambda c_G, c_el: solve_lp_upper(_Q_D, _P_D, c_G, c_el),
    "LP_approx": lambda c_G, c_el: solve_lp_approximated(_Q_D, _P_D, c_G, c_el, mode="mean_efficiency"),
}

TOLERANCES: dict[str, float] = {
    "MILP":      2e-2,
    "LP_lower":  1e-4,
    "LP_upper":  1e-4,
    "LP_approx": 1e-4,
}

_KEY_FMT = "cG={c_G:.4f}_cel={c_el:.4f}"


def _price_key(c_G: float, c_el: float) -> str:
    return _KEY_FMT.format(c_G=c_G, c_el=c_el)


# ---------------------------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------------------------
def generate_baselines() -> None:
    """Solve all formulations for all price pairs and write results to JSON."""
    baselines: dict = {
        "info": {
            "MILP_mip_gap": MILP_MIP_GAP,
        }
    }

    n_total = len(PRICE_PAIRS) * len(FORMULATIONS)
    done = 0
    for c_G, c_el in PRICE_PAIRS:
        key = _price_key(c_G, c_el)
        baselines[key] = {"c_G": c_G, "c_el": c_el}
        for name, solver in FORMULATIONS.items():
            done += 1
            print(f"[{done}/{n_total}] {name:12s}  cG={c_G:.4f}  cel={c_el:.4f} ... ", end="", flush=True)
            opex = solver(c_G, c_el)[0]
            baselines[key][name] = opex
            print(f"OPEX = {opex:,.2f}")

    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_FILE, "w") as fh:
        json.dump(baselines, fh, indent=2)
    print(f"\nBaselines written to {BASELINE_FILE}")


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------
def run_regression() -> bool:
    """Solve all formulations and compare to saved baselines. Returns True if all pass."""
    if not BASELINE_FILE.exists():
        print(f"ERROR: baseline file not found at {BASELINE_FILE}")
        print("Run with --generate first to create it.")
        return False

    with open(BASELINE_FILE) as fh:
        baselines: dict = json.load(fh)

    saved_gap = baselines.get("info", {}).get("MILP_mip_gap")
    if saved_gap != MILP_MIP_GAP:
        print(
            f"WARNING: baseline MIP gap ({saved_gap}) differs from "
            f"current ({MILP_MIP_GAP}). Consider re-running --generate."
        )

    results: list[dict] = []
    n_total = len(PRICE_PAIRS) * len(FORMULATIONS)
    done = 0

    for c_G, c_el in PRICE_PAIRS:
        key = _price_key(c_G, c_el)
        if key not in baselines:
            print(f"WARNING: no baseline for {key} -- skipping")
            continue
        baseline = baselines[key]

        for name, solver in FORMULATIONS.items():
            done += 1
            print(f"[{done}/{n_total}] {name:12s}  cG={c_G:.4f}  cel={c_el:.4f} ... ", end="", flush=True)
            opex = solver(c_G, c_el)[0]
            expected = baseline[name]
            tol = TOLERANCES[name]

            rel_err = abs(opex - expected) / max(abs(expected), 1.0)
            passed = rel_err <= tol
            status = "PASS" if passed else "FAIL"

            print(
                f"{status}  got={opex:,.2f}  expected={expected:,.2f}  "
                f"rel_err={rel_err:.2e}  tol={tol:.0e}"
            )
            results.append({"formulation": name, "c_G": c_G, "c_el": c_el,
                            "opex": opex, "expected": expected, "passed": passed})

    all_passed = all(r["passed"] for r in results)
    n_fail = sum(1 for r in results if not r["passed"])

    print()
    if all_passed:
        print(f"All {len(results)} regression checks PASSED.")
    else:
        print(f"{n_fail}/{len(results)} regression checks FAILED.")

    return all_passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    GENERATE = True
    if GENERATE:
        generate_baselines()
    else:
        ok = run_regression()
        sys.exit(0 if ok else 1)
