"""Commitment + TES dispatch plot and energy balance plot for the MILP, solved on the fly.

The MILP's δ are true binaries, so the commitment panel is Erdem's own on/off panel. Solves
the given ratio itself (via run_milp_dispatch.solve_and_save) and writes two files:
dispatch_milp_ratio<r>.pdf (commitment + TES) and energy_balance_milp_ratio<r>.pdf.

    python Marius/visualization/plot_milp_dispatch.py 2.0
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Marius" / "evaluation"))

from _dispatch_common import (
    dispatch_figure, energy_balance_figure, ratio_from_argv, save_figure,
)
from run_milp_dispatch import solve_and_save

FORMULATION = "milp"
DEFAULT_RATIO = 0.45


def main():
    ratio = ratio_from_argv(DEFAULT_RATIO)
    dispatch, _ = solve_and_save(ratio,5e-4)

    fig = dispatch_figure(dispatch, unit_panel="binary")
    save_figure(fig, f"dispatch_{FORMULATION}_ratio{ratio:.3f}")

    fig = energy_balance_figure(dispatch)
    save_figure(fig, f"energy_balance_{FORMULATION}_ratio{ratio:.3f}")


if __name__ == "__main__":
    main()
