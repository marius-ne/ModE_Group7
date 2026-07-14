"""Piecewise-linear part-load characteristics of the boiler and CHP, and how the LP
approximation's mean-efficiency line compares to them.

Moved out of Marius/notebooks/linearization.ipynb into a script. The piecewise-linear curves
are re-derived here as plain functions of the constants in Erdem/src/optimization/core.py,
since core.py only expresses the part-load curve inside a Pyomo constraint, not as a
standalone callable; the mean-efficiency slopes solve_lp_approximated(mode="mean_efficiency")
actually uses are imported from there directly.

Produces two figures in this directory:
  characteristics.pdf            the three piecewise-linear curves alone.
  lp_approx_characteristics.pdf  the same curves against the LP approximation's
                                  mean-efficiency line.

    python Marius/visualization/plot_linearization.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.optimization.core import (
    Q_out_nom_B, eta_nom_B, lambda_in_min_B, lambda_out_min_B, beta_B,
    Q_out_nom_CHP, eta_nom_CHP_th, eta_nom_CHP_el, P_out_nom_CHP,
    lambda_in_min_CHP, lambda_out_min_CHP_th, lambda_out_min_CHP_el, beta_CHP_th, beta_CHP_el,
    m_B_heat_mean, m_CHP_heat_mean, m_CHP_el_mean,
)
from src.visualization.style import apply_style

OUT_DIR = ROOT / "Marius" / "visualization"


def boiler_heat_output(qin: float) -> float:
    """Boiler thermal output for fuel input qin [kW]: the same piecewise-linear part-load
    curve as build_milp's boiler heat constraint (Erdem/src/optimization/core.py)."""
    qin_min = lambda_in_min_B * Q_out_nom_B / eta_nom_B
    if qin < qin_min:
        return 0.0
    qin = min(qin, Q_out_nom_B / eta_nom_B)
    return Q_out_nom_B * (
        lambda_out_min_B + (1.0 / beta_B) * (qin * eta_nom_B / Q_out_nom_B - lambda_in_min_B)
    )


def chp_heat_output(qin: float) -> float:
    """CHP thermal output for fuel input qin [kW]: the same piecewise-linear part-load curve
    as build_milp's CHP heat constraint."""
    qin_min = lambda_in_min_CHP * Q_out_nom_CHP / eta_nom_CHP_th
    if qin < qin_min:
        return 0.0
    qin = min(qin, Q_out_nom_CHP / eta_nom_CHP_th)
    return Q_out_nom_CHP * (
        lambda_out_min_CHP_th
        + (1.0 / beta_CHP_th) * (qin * eta_nom_CHP_th / Q_out_nom_CHP - lambda_in_min_CHP)
    )


def chp_electricity_output(qin: float) -> float:
    """CHP electrical output for fuel input qin [kW]: the same piecewise-linear part-load
    curve as build_milp's CHP electricity constraint. Capped at the same fuel input as the
    heat side, since both outputs are driven by one fuel input."""
    qin_min = lambda_in_min_CHP * P_out_nom_CHP / eta_nom_CHP_el
    if qin < qin_min:
        return 0.0
    qin = min(qin, Q_out_nom_CHP / eta_nom_CHP_th)
    return P_out_nom_CHP * (
        lambda_out_min_CHP_el
        + (1.0 / beta_CHP_el) * (qin * eta_nom_CHP_el / P_out_nom_CHP - lambda_in_min_CHP)
    )


def plot_characteristics_alone(x_boiler: np.ndarray, x_chp: np.ndarray) -> plt.Figure:
    """The three piecewise-linear curves alone, with no linear comparison."""
    fig, axes = plt.subplots(1, 3, constrained_layout=True)

    axes[0].plot(x_boiler, [boiler_heat_output(x) for x in x_boiler], linewidth=2, color="C0")
    axes[0].set_xlabel(r"$\dot{Q}^{\,\mathrm{in}}_{\mathrm{B}}$ [kW]")
    axes[0].set_ylabel(r"$\dot{Q}^{\,\mathrm{out}}_{\mathrm{B}}$ [kW]")

    axes[1].plot(x_chp, [chp_heat_output(x) for x in x_chp], linewidth=2, color="C0")
    axes[1].set_xlabel(r"$\dot{Q}^{\,\mathrm{in}}_{\mathrm{CHP}}$ [kW]")
    axes[1].set_ylabel(r"$\dot{Q}^{\,\mathrm{out}}_{\mathrm{CHP}}$ [kW]")

    axes[2].plot(x_chp, [chp_electricity_output(x) for x in x_chp], linewidth=2, color="C0")
    axes[2].set_xlabel(r"$\dot{Q}^{\,\mathrm{in}}_{\mathrm{CHP}}$ [kW]")
    axes[2].set_ylabel(r"$\dot{P}^{\,\mathrm{out}}_{\mathrm{CHP}}$ [kW]")

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)

    return fig


def plot_characteristics_vs_lp_approx(x_boiler: np.ndarray, x_chp: np.ndarray) -> plt.Figure:
    """The piecewise-linear curves against the LP approximation's mean-efficiency line --
    i.e. what solve_lp_approximated(mode="mean_efficiency") assumes instead of the real
    part-load curve."""
    fig, axes = plt.subplots(1, 3, constrained_layout=True)

    axes[0].plot(x_boiler, [boiler_heat_output(x) for x in x_boiler],
                label="Piecewise-linear", linewidth=2, color="C0")
    axes[0].plot(x_boiler, m_B_heat_mean * x_boiler,
                label=f"LP approx mean-eff (m={m_B_heat_mean:.4f})",
                linestyle="--", color="C1", linewidth=2)
    axes[0].set_xlabel(r"$\dot{Q}^{\,\mathrm{in}}_{\mathrm{B}}$ [kW]")
    axes[0].set_ylabel(r"$\dot{Q}^{\,\mathrm{out}}_{\mathrm{B}}$ [kW]")

    axes[1].plot(x_chp, [chp_heat_output(x) for x in x_chp],
                label="Piecewise-linear", linewidth=2, color="C0")
    axes[1].plot(x_chp, m_CHP_heat_mean * x_chp,
                label=f"LP approx mean-eff (m={m_CHP_heat_mean:.4f})",
                linestyle="--", color="C1", linewidth=2)
    axes[1].set_xlabel(r"$\dot{Q}^{\,\mathrm{in}}_{\mathrm{CHP}}$ [kW]")
    axes[1].set_ylabel(r"$\dot{Q}^{\,\mathrm{out}}_{\mathrm{CHP}}$ [kW]")

    axes[2].plot(x_chp, [chp_electricity_output(x) for x in x_chp],
                label="Piecewise-linear", linewidth=2, color="C0")
    axes[2].plot(x_chp, m_CHP_el_mean * x_chp,
                label=f"LP approx mean-eff (m={m_CHP_el_mean:.4f})",
                linestyle="--", color="C1", linewidth=2)
    axes[2].set_xlabel(r"$\dot{Q}^{\,\mathrm{in}}_{\mathrm{CHP}}$ [kW]")
    axes[2].set_ylabel(r"$\dot{P}^{\,\mathrm{out}}_{\mathrm{CHP}}$ [kW]")

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
        top = ax.get_ylim()[1]
        ax.set_ylim(top=top * 1.3)
        ax.legend(loc="upper left")

    return fig


def main():
    apply_style(width_cm=24, aspect=3.9, ncols=3, grid=False, strict=True)

    x_boiler = np.linspace(0.0, Q_out_nom_B / eta_nom_B, 300)
    x_chp = np.linspace(0.0, Q_out_nom_CHP / eta_nom_CHP_th, 300)

    fig = plot_characteristics_alone(x_boiler, x_chp)
    out = OUT_DIR / "characteristics.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out}")

    fig = plot_characteristics_vs_lp_approx(x_boiler, x_chp)
    out = OUT_DIR / "lp_approx_characteristics.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
