"""Commitment + TES dispatch plot and energy balance plot for the LP lower bound, solved on
the fly.

The LP lower bound is the binary relaxation, so its δ live anywhere in [0, 1] and the
commitment panel is the continuous one. Solves the given ratio itself (via
run_lp_lower_dispatch.solve_and_save) and writes two files: dispatch_lp_lower_ratio<r>.pdf
(commitment + TES) and energy_balance_lp_lower_ratio<r>.pdf.

    python Marius/visualization/plot_lp_lower_dispatch.py 1.0
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Marius" / "evaluation"))

from _dispatch_common import (
    dispatch_figure, energy_balance_figure, ratio_from_argv, save_figure, title_of,
)
from run_lp_lower_dispatch import solve_and_save

FORMULATION = "lp_lower"
DEFAULT_RATIO = 0.5


def main():
    ratio = ratio_from_argv(DEFAULT_RATIO)
    dispatch, meta = solve_and_save(ratio)

    fig = dispatch_figure(dispatch, unit_panel="relaxed")
    save_figure(fig, f"dispatch_{FORMULATION}_ratio{ratio:.3f}")

    fig = energy_balance_figure(dispatch, title_of("LP lower bound — energy balance", meta))
    save_figure(fig, f"energy_balance_{FORMULATION}_ratio{ratio:.3f}")


if __name__ == "__main__":
    main()
