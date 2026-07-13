"""TES dispatch plot and energy balance plot for the LP approximation, solved on the fly.

build_lp_approximated drops the commitment variables from the model altogether (that is what
makes it an approximation rather than a bound), so there is no δ to draw and no delta plot for
this formulation -- the dispatch figure is the TES panel alone. Every formulation still has an
energy balance. Solves the given ratio itself (via run_lp_approx_dispatch.solve_and_save) and
writes two files: dispatch_lp_approx_ratio<r>.pdf (TES only) and
energy_balance_lp_approx_ratio<r>.pdf.

    python Marius/visualization/plot_lp_approx_dispatch.py 1.0
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Marius" / "evaluation"))

from _dispatch_common import (
    dispatch_figure, energy_balance_figure, ratio_from_argv, save_figure,
)
from run_lp_approx_dispatch import solve_and_save

FORMULATION = "lp_approx"
DEFAULT_RATIO = 0.5


def main():
    ratio = ratio_from_argv(DEFAULT_RATIO)
    dispatch, _ = solve_and_save(ratio)

    fig = dispatch_figure(dispatch, unit_panel=None)
    save_figure(fig, f"dispatch_{FORMULATION}_ratio{ratio:.3f}")

    fig = energy_balance_figure(dispatch)
    save_figure(fig, f"energy_balance_{FORMULATION}_ratio{ratio:.3f}")


if __name__ == "__main__":
    main()
