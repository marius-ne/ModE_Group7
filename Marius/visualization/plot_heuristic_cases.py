"""Abstract schematic of chp_on and boilers_on heuristic decision cases."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.visualization.style import apply_style

_ROMAN = ["I", "II", "III", "IV", "V"]
_D     = 1.0

_GREEN = "#2E7D32"
_RED   = "#D6604D"
_ORG   = "#F4A261"
_NR_C  = "#D0D0D0"
_LINE  = "#222222"
_1OR2  = r"1\ \mathrm{or}\ 2"
_LEAD   = 0.10   # length of the dashed leader from a bar edge to its bound label
_BAR_DX = -0.26  # bar centre offset from the column centre

# Vertical positions of the rows below the axis, in axes fractions
_Y_CASE = -0.015
_Y_ELSE = -0.16
_Y_DEC  = -0.28
_Y_ROW1 = -0.40
_Y_ROW2 = -0.56


def _b(lo, hi, col, lbl):
    return (lo, hi, col, lbl)


# Bar spans, chosen so the Q_min / Q_max edge labels clear the demand line
_SPAN_OK   = (0.30, 1.80)   # straddles demand
_SPAN_HIGH = (1.26, 1.82)   # min above demand (overproduces)
_SPAN_LOW  = (0.30, 0.80)   # max below demand (insufficient)

# CHP mode: heat and power trigger conditions
_C_CHP_HEAT = [
    _b(*_SPAN_OK,   _GREEN, "2\nCHP"),
    _b(*_SPAN_HIGH, _RED,   "2\nCHP"),
    _b(*_SPAN_LOW,  _ORG,   "2\nCHP"),
    _b(*_SPAN_LOW,  _ORG,   "1\nCHP"),
    _b(*_SPAN_HIGH, _RED,   "1\nCHP"),
]
_C_CHP_POWER = [
    _b(*_SPAN_OK,   _GREEN, "2\nCHP"),
    _b(*_SPAN_HIGH, _RED,   "2\nCHP"),
    None,                               # Case III: heat-only condition
    None,                               # Case IV: heat-only condition
    _b(*_SPAN_HIGH, _RED,   "1\nCHP"),
]

# Boiler mode: heat trigger condition only
_B_BOI_HEAT = [
    _b(*_SPAN_OK,   _GREEN, "2 B"),
    _b(*_SPAN_HIGH, _RED,   "2 B"),
    _b(*_SPAN_LOW,  _ORG,   "2 B"),
    _b(*_SPAN_LOW,  _ORG,   "1 B"),
    _b(*_SPAN_HIGH, _RED,   "1 B"),
]

# Delta outcomes per case I–V: (primary_sum, secondary_sum)
# CHP mode: primary=CHP, secondary=B
_CHP_DELTAS = [(2, 0), (1, 0), (2, _1OR2), (1, _1OR2), (0, _1OR2)]
# Boiler mode: primary=B, secondary=CHP
_BOI_DELTAS = [(2, 0), (1, 0), (2, _1OR2), (1, _1OR2), (0, _1OR2)]


def _draw_bars(ax, bars, sym, deltas, prim_lbl, sec_lbl, fontsize,
               show_delta_labels=False):
    x  = np.arange(5)          # column centres
    bx = x + _BAR_DX           # bar centres, offset left so the bound labels stay in-column
    w = 0.38
    fs_edge = fontsize + 1

    d_label   = f"${sym}_{{\\mathrm{{Demand}},k}}$"
    max_label = f"${sym}_{{\\mathrm{{max}}}}$"
    min_label = f"${sym}_{{\\mathrm{{min}}}}$"

    for i in range(0, 5, 2):
        ax.axvspan(i - 0.5, i + 0.5, facecolor="#EBEBEB", alpha=1.0, zorder=0)

    for i, bar in enumerate(bars):
        if bar is None:
            ax.bar(bx[i], 0.12, bottom=_D - 0.06, width=w,
                   color=_NR_C, alpha=0.8, edgecolor="#888888",
                   linewidth=0.5, linestyle=":", zorder=1)
            continue

        lo, hi, color, label = bar
        ax.bar(bx[i], hi - lo, bottom=lo, width=w,
               color=color, edgecolor="black", linewidth=0.8, zorder=1)

        # Unit count, at the top inside the bar
        tc = "white" if color in (_GREEN, _RED) else "#333333"
        ax.text(bx[i], hi - 0.06, label, ha="center", va="top",
                fontsize=fontsize - 2, color=tc, fontweight="bold", zorder=2)

        # Bar edges = the operating bounds of the active unit(s); dashed leaders
        # carry each edge out to its label on the right of the bar
        for y, edge_label in ((hi, max_label), (lo, min_label)):
            ax.plot([bx[i] + w / 2, bx[i] + w / 2 + _LEAD], [y, y],
                    linestyle="--", linewidth=0.8, color=_LINE, zorder=2)
            ax.text(bx[i] + w / 2 + _LEAD + 0.04, y, edge_label,
                    ha="left", va="center", fontsize=fs_edge, color=_LINE, zorder=2)

    ax.axhline(_D, color=_LINE, linestyle="--", linewidth=1.6, zorder=3)
    # Demand label to the LEFT of the subplot
    ax.text(-0.72, _D, d_label, ha="right", va="center",
            fontsize=fontsize+2, color=_LINE, fontweight="bold", clip_on=False)

    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(_ROMAN, fontsize=fontsize + 2, fontweight="bold")
    ax.tick_params(axis="x", length=0)
    ax.minorticks_off()
    ax.set_xlim(-0.65, 4.65)
    ax.set_ylim(0.0, 2.15)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if deltas is None:
        return

    # Blended transform: x in data coords, y in axes fraction
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    fs_val   = fontsize + 2
    fs_label = fontsize + 3

    # The cases form an if/elif chain: each one is only reached if all earlier
    # ones failed.  Arrows between the columns carry that fall-through.
    for i in range(4):
        ax.annotate("", xy=(x[i + 1] - 0.22, _Y_ELSE), xytext=(x[i] + 0.22, _Y_ELSE),
                    xycoords=trans, textcoords=trans, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", linewidth=0.9,
                                    color="#777777", shrinkA=0, shrinkB=0))
        ax.text(x[i] + 0.5, _Y_ELSE + 0.015, "else", transform=trans,
                ha="center", va="bottom", fontsize=fontsize - 1, style="italic",
                color="#777777", clip_on=False)

    # Row 1: primary delta values
    for i, (pv, _sv) in enumerate(deltas):
        ax.text(x[i], _Y_ROW1, f"${pv}$", transform=trans,
                ha="center", va="baseline", fontsize=fs_val, color="#333333", clip_on=False)

    # Row 2: secondary delta values
    for i, (_pv, sv) in enumerate(deltas):
        ax.text(x[i], _Y_ROW2, f"${sv}$", transform=trans,
                ha="center", va="baseline", fontsize=fs_val, color="#333333", clip_on=False)

    if show_delta_labels:
        l1 = f"$\\sum_{{i}}\\,\\delta_{{\\mathrm{{{prim_lbl}}},i,k}}=$"
        l2 = f"$\\sum_{{i}}\\,\\delta_{{\\mathrm{{{sec_lbl}}},i,k}}=$"
        # Same top edge as the roman-numeral tick labels (their pad, in axes fractions)
        ax.text(-0.5, _Y_CASE, "Cases:", transform=trans,
                ha="right", va="top", fontsize=fontsize + 2, color="#333333",
                fontweight="bold", clip_on=False)
        ax.text(-0.65, _Y_DEC, "Decisions:", transform=trans,
                ha="right", va="baseline", fontsize=fontsize + 2, color="#333333",
                fontweight="bold", clip_on=False)
        ax.text(-0.65, _Y_ROW1, l1, transform=trans,
                ha="right", va="baseline", fontsize=fs_label, color="#333333", clip_on=False)
        ax.text(-0.65, _Y_ROW2, l2, transform=trans,
                ha="right", va="baseline", fontsize=fs_label, color="#333333", clip_on=False)


def _make_figure(bar_sets, syms, col_titles, deltas, prim_lbl, sec_lbl,
                 title, fontsize, width_cm, aspect):
    n = len(bar_sets)
    apply_style(width_cm=width_cm, aspect=aspect, strict=True)
    fig, axes = plt.subplots(1, n)
    if n == 1:
        axes = [axes]

    for j, (ax, bars, sym, col_title) in enumerate(
        zip(axes, bar_sets, syms, col_titles)
    ):
        is_first = (j == 0)
        _draw_bars(ax, bars, sym,
                   deltas=deltas,
                   prim_lbl=prim_lbl, sec_lbl=sec_lbl,
                   fontsize=fontsize,
                   show_delta_labels=is_first)
        if col_title is not None:
            ax.set_title(col_title, fontsize=fontsize, pad=8, fontweight="bold")

    # Without column titles there is nothing to fill the band under the suptitle
    top = 0.84 if any(t is not None for t in col_titles) else 0.92

    fig.suptitle(title, fontsize=fontsize + 2)
    fig.subplots_adjust(bottom=0.40, top=top, wspace=0.22)
    return fig


def plot_heuristic_cases(
    output_dir: str | None = None,
    fontsize: int = 9,
) -> tuple[Path, Path]:
    base = Path(output_dir) if output_dir else Path("Marius/visualization")
    base.mkdir(parents=True, exist_ok=True)

    chp_fig = _make_figure(
        bar_sets=[_C_CHP_HEAT, _C_CHP_POWER],
        syms=[r"\dot{Q}", "P"],
        col_titles=["CHP heat output", "CHP electrical output"],
        deltas=_CHP_DELTAS,
        prim_lbl="CHP",
        sec_lbl="B",
        title=r"$LP^{U}$ CHP mode heuristics",
        fontsize=fontsize,
        width_cm=32,
        aspect=2.4,
    )

    boi_fig = _make_figure(
        bar_sets=[_B_BOI_HEAT],
        syms=[r"\dot{Q}"],
        col_titles=[None],
        deltas=_BOI_DELTAS,
        prim_lbl="B",
        sec_lbl="CHP",
        title=r"$LP^{U}$ Boiler mode heuristics: boiler heat output",
        fontsize=fontsize,
        width_cm=17,
        aspect=1.4,
    )

    chp_path = base / "lp_upper_heuristic_cases_chp.png"
    boi_path = base / "lp_upper_heuristic_cases_boiler.png"

    chp_fig.savefig(chp_path, bbox_inches="tight", pad_inches=0.3)
    boi_fig.savefig(boi_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(chp_fig)
    plt.close(boi_fig)

    return chp_path, boi_path


if __name__ == "__main__":
    chp_out, boi_out = plot_heuristic_cases()
    print(f"CHP figure    -> {chp_out}")
    print(f"Boiler figure -> {boi_out}")
