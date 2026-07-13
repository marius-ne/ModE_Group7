"""Commitment + TES dispatch plot and energy balance plot for the LP upper bound, solved on
the fly.

The LP upper bound fixes δ to a heuristic schedule rather than optimizing it, so its δ are
given, not solved for -- but they are still true 0/1 values, so the commitment panel is
Erdem's own binary on/off panel, same as the MILP's. Solves the given ratio itself (via
run_lp_upper_dispatch.solve_and_save) and writes two files: dispatch_lp_upper_ratio<r>.pdf
(commitment + TES) and energy_balance_lp_upper_ratio<r>.pdf.

    python Marius/visualization/plot_lp_upper_dispatch.py 1.0
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Marius" / "evaluation"))

from _dispatch_common import (
    dispatch_figure, energy_balance_figure, ratio_from_argv, save_figure,
)
from run_lp_upper_dispatch import solve_and_save

FORMULATION = "lp_upper"
DEFAULT_RATIO = 0.5


def main():
    ratio = ratio_from_argv(DEFAULT_RATIO)
    dispatch, _ = solve_and_save(ratio)

    fig = dispatch_figure(dispatch, unit_panel="binary")
    save_figure(fig, f"dispatch_{FORMULATION}_ratio{ratio:.3f}")

    fig = energy_balance_figure(dispatch)
    save_figure(fig, f"energy_balance_{FORMULATION}_ratio{ratio:.3f}")


if __name__ == "__main__":
    main()
