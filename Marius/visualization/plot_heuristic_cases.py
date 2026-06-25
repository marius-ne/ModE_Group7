"""Abstract schematic of chp_on and boilers_on heuristic decision cases."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

_ROMAN = ["I", "II", "III", "IV", "V"]
_D     = 1.0

_GREEN = "#2E7D32"
_RED   = "#D6604D"
_ORG   = "#F4A261"
_NR_C  = "#D0D0D0"
_LINE  = "#222222"
_HALF  = "1/2"


def _b(lo, hi, col, lbl):
    return (lo, hi, col, lbl)


# CHP mode: heat and power trigger conditions
# II & V share (1.15, 1.75): bar starts above demand line (min overproduces)
# III & IV share (0.25, 0.85): bar ends below demand line (max insufficient)
_C_CHP_HEAT = [
    _b(0.25, 1.75, _GREEN, "2\nCHP"),
    _b(1.15, 1.75, _RED,   "2\nCHP"),
    _b(0.25, 0.85, _ORG,   "2\nCHP"),
    _b(0.25, 0.85, _ORG,   "1\nCHP"),
    _b(1.15, 1.75, _RED,   "1\nCHP"),
]
_C_CHP_POWER = [
    _b(0.25, 1.75, _GREEN, "2\nCHP"),
    _b(1.15, 1.75, _RED,   "2\nCHP"),
    None,                               # Case III: heat-only condition
    None,                               # Case IV: heat-only condition
    _b(1.15, 1.75, _RED,   "1\nCHP"),
]

# Boiler mode: heat trigger condition only
_B_BOI_HEAT = [
    _b(0.25, 1.75, _GREEN, "2 B"),
    _b(1.15, 1.75, _RED,   "2 B"),
    _b(0.25, 0.85, _ORG,   "2 B"),
    _b(0.25, 0.85, _ORG,   "1 B"),
    _b(1.15, 1.75, _RED,   "1 B"),
]

# Delta outcomes per case I–V: (primary_sum, secondary_sum)
# CHP mode: primary=CHP, secondary=B
_CHP_DELTAS = [(2, 0), (1, 0), (2, _HALF), (1, _HALF), (0, _HALF)]
# Boiler mode: primary=B, secondary=CHP
_BOI_DELTAS = [(2, 0), (1, 0), (2, _HALF), (1, _HALF), (0, _HALF)]


def _draw_bars(ax, bars, d_label, deltas, prim_lbl, sec_lbl, fontsize,
               show_delta_labels=False):
    x = np.arange(5)
    w = 0.52

    for i in range(0, 5, 2):
        ax.axvspan(i - 0.48, i + 0.48, facecolor="#EBEBEB", alpha=1.0, zorder=0)

    for i, bar in enumerate(bars):
        if bar is None:
            ax.bar(x[i], 0.12, bottom=_D - 0.06, width=w,
                   color=_NR_C, alpha=0.8, edgecolor="#888888",
                   linewidth=0.5, linestyle=":", zorder=1)
        else:
            lo, hi, color, label = bar
            ax.bar(x[i], hi - lo, bottom=lo, width=w,
                   color=color, alpha=0.88, edgecolor="black", linewidth=0.8, zorder=1)
            tc = "white" if color in (_GREEN, _RED) else "#333333"
            ax.text(x[i], hi - 0.06, label,
                    ha="center", va="top",
                    fontsize=fontsize - 2, color=tc, fontweight="bold", zorder=2)

    ax.axhline(_D, color=_LINE, linestyle="--", linewidth=1.6, zorder=3)
    # Demand label to the LEFT of the subplot
    ax.text(-0.72, _D, d_label, ha="right", va="center",
            fontsize=fontsize+2, color=_LINE, fontweight="bold", clip_on=False)

    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(_ROMAN, fontsize=fontsize + 2, fontweight="bold")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.65, 4.65)
    ax.set_ylim(0.0, 2.1)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if deltas is None:
        return

    # Blended transform: x in data coords, y in axes fraction
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    fs_val   = fontsize + 2
    fs_label = fontsize + 5

    # Row 1: primary delta values
    for i, (pv, _sv) in enumerate(deltas):
        ax.text(x[i], -0.18, f"${pv}$", transform=trans,
                ha="center", va="baseline", fontsize=fs_val, color="#333333", clip_on=False)

    # Row 2: secondary delta values
    for i, (_pv, sv) in enumerate(deltas):
        ax.text(x[i], -0.36, f"${sv}$", transform=trans,
                ha="center", va="baseline", fontsize=fs_val, color="#333333", clip_on=False)

    if show_delta_labels:
        l1 = f"$\\sum_{i}\\delta_{{\\mathrm{{{prim_lbl}}},i,k}}=$"
        l2 = f"$\\sum_{i}\\delta_{{\\mathrm{{{sec_lbl}}},i,k}}=$"
        ax.text(-0.65, -0.18, l1, transform=trans,
                ha="right", va="baseline", fontsize=fs_label, color="#333333", clip_on=False)
        ax.text(-0.65, -0.36, l2, transform=trans,
                ha="right", va="baseline", fontsize=fs_label, color="#333333", clip_on=False)


def _make_figure(bar_sets, d_labels, col_titles, deltas, prim_lbl, sec_lbl,
                 title, fontsize):
    n = len(bar_sets)
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 4, 5))
    if n == 1:
        axes = [axes]

    for j, (ax, bars, d_lbl, col_title) in enumerate(
        zip(axes, bar_sets, d_labels, col_titles)
    ):
        is_first = (j == 0)
        _draw_bars(ax, bars, d_lbl,
                   deltas=deltas,
                   prim_lbl=prim_lbl, sec_lbl=sec_lbl,
                   fontsize=fontsize,
                   show_delta_labels=is_first)
        ax.set_title(col_title, fontsize=fontsize, pad=8, fontweight="bold")

    fig.suptitle(title, fontsize=fontsize + 4)
    fig.subplots_adjust(bottom=0.34, top=0.85, wspace=0.22)
    return fig


def plot_heuristic_cases(
    output_dir: str | None = None,
    fontsize: int = 11,
) -> tuple[Path, Path]:
    base = Path(output_dir) if output_dir else Path("Marius/visualization")
    base.mkdir(parents=True, exist_ok=True)

    chp_fig = _make_figure(
        bar_sets=[_C_CHP_HEAT, _C_CHP_POWER],
        d_labels=[r"$\dot{Q}_D$", r"$P_D$"],
        col_titles=["CHP heat output", "CHP electrical output"],
        deltas=_CHP_DELTAS,
        prim_lbl="CHP",
        sec_lbl="B",
        title=r"$LP^{U}$ CHP mode heuristics",
        fontsize=fontsize+2,
    )

    boi_fig = _make_figure(
        bar_sets=[_B_BOI_HEAT],
        d_labels=[r"$\dot{Q}_D$"],
        col_titles=["Boiler heat output"],
        deltas=_BOI_DELTAS,
        prim_lbl="B",
        sec_lbl="CHP",
        title=r"$LP^{U}$ Boiler mode heuristics",
        fontsize=fontsize,
    )

    chp_path = base / "lp_upper_heuristic_cases_chp.png"
    boi_path = base / "lp_upper_heuristic_cases_boiler.png"

    chp_fig.savefig(chp_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    boi_fig.savefig(boi_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(chp_fig)
    plt.close(boi_fig)

    return chp_path, boi_path


if __name__ == "__main__":
    chp_out, boi_out = plot_heuristic_cases()
    print(f"CHP figure    → {chp_out}")
    print(f"Boiler figure → {boi_out}")
