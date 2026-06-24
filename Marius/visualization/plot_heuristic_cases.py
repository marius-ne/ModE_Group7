"""Three-panel bar chart illustrating the chp_on heuristic output capacity ranges."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.append("Erdem")
from src.optimization.core import (
    Q_out_nom_B, lambda_out_min_B,
    Q_out_nom_CHP, P_out_nom_CHP,
    lambda_out_min_CHP_th, lambda_out_min_CHP_el,
)

_B_min   = Q_out_nom_B   * lambda_out_min_B
_B_max   = Q_out_nom_B
_CQ_min  = Q_out_nom_CHP * lambda_out_min_CHP_th
_CQ_max  = Q_out_nom_CHP
_CP_min  = P_out_nom_CHP * lambda_out_min_CHP_el
_CP_max  = P_out_nom_CHP

# Combined output range [min, max] per case I–V
#   I:   B1,            no CHPs
#   II:  CHP1+CHP2,     no boilers
#   III: CHP1,          no boilers
#   IV:  CHP1 + B1
#   V:   CHP1 + B1 + B2
_B_LO  = np.array([_B_min,      0.0,          0.0,         _B_min,      2*_B_min])
_B_HI  = np.array([_B_max,      0.0,          0.0,         _B_max,      2*_B_max])
_CQ_LO = np.array([0.0,         2*_CQ_min,    _CQ_min,     _CQ_min,     _CQ_min])
_CQ_HI = np.array([0.0,         2*_CQ_max,    _CQ_max,     _CQ_max,     _CQ_max])
_CP_LO = np.array([0.0,         2*_CP_min,    _CP_min,     _CP_min,     _CP_min])
_CP_HI = np.array([0.0,         2*_CP_max,    _CP_max,     _CP_max,     _CP_max])

_C_FEASIBLE   = "#4DAC26"
_C_INFEASIBLE = "#D6604D"
_C_ZERO       = "#CCCCCC"
_C_DEMAND     = "#111111"
_ROMAN        = ["I", "II", "III", "IV", "V"]


def _bar_colors(lo, hi, D):
    out = []
    for a, b in zip(lo, hi):
        if b == 0.0:
            out.append(_C_ZERO)
        elif a > D:
            out.append(_C_INFEASIBLE)
        else:
            out.append(_C_FEASIBLE)
    return out


def plot_heuristic_cases(
    D_B:  float = 350.0,
    D_CH: float = 350.0,
    D_CE: float = 220.0,
    output_path: str | None = None,
    fontsize: int = 11,
) -> Path:
    """Three-panel bar chart of output capacity ranges for the five chp_on cases.

    Left:   boiler heat output [kW]
    Middle: CHP heat output [kW]
    Right:  CHP electrical output [kW]

    Each bar spans [min_combined, max_combined] for the committed units in that
    case.  A constant demand line D shows whether a configuration is feasible
    (D inside the bar) or infeasible (min > D, bar entirely above the line).

    Parameters
    ----------
    D_B:  heat demand reference line for the boiler subplot [kW]
    D_CH: heat demand reference line for the CHP heat subplot [kW]
    D_CE: electrical demand reference line for the CHP elec subplot [kW]
    """
    if output_path is None:
        output_path = "Marius/visualization/lp_upper_heuristic_cases.png"

    x = np.arange(5)
    w = 0.55

    panels = [
        ("Boiler heat output [kW]",      _B_LO,  _B_HI,  D_B),
        ("CHP heat output [kW]",         _CQ_LO, _CQ_HI, D_CH),
        ("CHP electrical output [kW]",   _CP_LO, _CP_HI, D_CE),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, (title, lo, hi, D) in zip(axes, panels):
        colors = _bar_colors(lo, hi, D)

        for i in range(5):
            if hi[i] > 0.0:
                ax.bar(
                    x[i], hi[i] - lo[i], bottom=lo[i],
                    width=w, color=colors[i], alpha=0.82,
                    edgecolor="black", linewidth=0.8,
                )
            else:
                ax.bar(
                    x[i], 0.8, bottom=0,
                    width=w, color=_C_ZERO, alpha=0.55,
                    edgecolor="black", linewidth=0.6, linestyle=":",
                )

        ax.axhline(D, color=_C_DEMAND, linestyle="--", linewidth=1.8,
                   label=f"Demand = {D:.0f} kW", zorder=5)

        ax.set_title(title, fontsize=fontsize, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(_ROMAN, fontsize=fontsize + 1, fontweight="bold")
        ax.set_xlabel("Heuristic case", fontsize=fontsize)
        ax.set_ylabel("Output [kW]", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1)
        ax.legend(fontsize=fontsize - 1, loc="upper right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax.set_xlim(-0.65, 4.65)
        ax.set_ylim(bottom=0)

    legend_handles = [
        mpatches.Patch(facecolor=_C_FEASIBLE,   edgecolor="black", alpha=0.85,
                       label=r"Feasible: $Q_\mathrm{min} \leq D \leq Q_\mathrm{max}$"),
        mpatches.Patch(facecolor=_C_INFEASIBLE, edgecolor="black", alpha=0.85,
                       label=r"Infeasible: $Q_\mathrm{min} > D$ → unit switched off"),
        mpatches.Patch(facecolor=_C_ZERO,       edgecolor="black", alpha=0.65,
                       label=r"Not committed ($Q_\mathrm{max} = 0$)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=fontsize - 1, framealpha=0.90,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        r"LP Upper Bound — $\mathtt{chp\_on}$ heuristic: combined output capacity per case (I–V)",
        fontsize=fontsize + 1, y=1.01,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

if __name__ == "__main__":
    out = plot_heuristic_cases(D_B=350.0, D_CH=350.0, D_CE=220.0)
    print(f"Heuristic figure saved → {out}")
    
