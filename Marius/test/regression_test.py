"""
Regression test suite for the three Marius dispatch formulations.

Usage
-----
  # (Re-)generate baselines from scratch and save to file:
  python Marius/regression_test.py --generate

  # Check current formulations against saved baselines:
  python Marius/regression_test.py

The baseline file is stored at Marius/results/regression_baselines.json.
Each entry records OPEX (total cost) for a (c_G, c_el) price pair.

Tolerances
----------
  MILP        : 2e-3 relative  (MIPGap=1e-3 leaves headroom for branch variability)
  LP          : 1e-4 relative  (pure LP, deterministic up to numerical noise)
  LP_binary   : 1e-4 relative  (pure LP, deterministic up to numerical noise)
"""

import json
import sys
from pathlib import Path

# Ensure Marius/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import formulation_MILP as _milp
import formulation_LP as _lp
import formulation_LP_binary as _lp_bin

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASELINE_FILE = Path(__file__).parent / "results" / "regression_baselines.json"

# Five price pairs spanning the interesting range of gas vs electricity costs.
# Each triggers a meaningfully different dispatch strategy.
PRICE_PAIRS: list[tuple[float, float]] = [
    (0.08, 0.08),   # both cheap: CHPs run flat-out, minimal grid
    (0.10, 0.25),   # cheap gas, moderate electricity: CHPs preferred
    (0.16, 0.30),   # mid-range: balanced dispatch
    (0.22, 0.12),   # expensive gas, cheap electricity: more grid import
    (0.28, 0.50),   # both expensive: tight optimisation
]

# Hyperparameters recorded in the baseline "info" section
MILP_MIP_GAP: float = 1e-3

def _milp_solve(c_G: float, c_el: float) -> tuple[float, object]:
    return _milp.solve(c_G, c_el, mip_gap=MILP_MIP_GAP)

FORMULATIONS: dict[str, callable] = {
    "MILP":      _milp_solve,
    "LP":        _lp.solve,
    "LP_binary": _lp_bin.solve,
}

# Relative tolerance per formulation
TOLERANCES: dict[str, float] = {
    "MILP":      2e-3,
    "LP":        1e-4,
    "LP_binary": 1e-4,
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
            opex, _ = solver(c_G, c_el)
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
            opex, _ = solver(c_G, c_el)
            expected = baseline[name]
            tol = TOLERANCES[name]

            # Relative error; guard against near-zero expected values
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
    if "--generate" in sys.argv:
        generate_baselines()
    else:
        ok = run_regression()
        sys.exit(0 if ok else 1)
