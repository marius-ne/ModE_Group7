"""Two-panel illustration of why angle-ratio sampling extrapolates further than
plain 2D LHS sampling. Left: the feasible c_el/c_G price rectangle with the 40
LHS training samples (as in Erdem's sampling). Right: the same rectangle with
only its LR (min-ratio) and UL (max-ratio) corner rays drawn from the origin,
and the convex cone spanned by those rays shaded -- illustrating that a model
trained on the ratio alone generalizes to any price pair on that cone, a far
larger region than the original rectangle."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from pathlib import Path

sys.path.append("Erdem")
from src.sampling.core import GAS_MIN, GAS_MAX, ELEC_MIN, ELEC_MAX, create_sample

_BOUND_COLOR = "#888780"
_BLUE = "#2166AC"
_CONE_FACE = "#D6604D"
_CONE_EDGE = "#B2182B"
_SLOPE_COLOR = "#B2182B"

N_LHS = 40
LEFT_PAD_FRAC = 0.25    # padding around the rectangle in the left panel's x-axis
Y_MAX_LEFT = 400.0       # left panel's y-axis extent
Y_MAX_RIGHT = 600.0      # right panel's y-axis extent

FIG_WIDTH = 15.0
GRID_LEFT, GRID_RIGHT, GRID_TOP, GRID_BOTTOM, WSPACE = 0.06, 0.97, 0.88, 0.11, 0.3


def _draw_rectangle(ax, shaded: bool = True, label: str | None = None):
    for x in (ELEC_MIN, ELEC_MAX):
        ax.axvline(x, color=_BOUND_COLOR, linewidth=1, linestyle="--", zorder=1)
    for y in (GAS_MIN, GAS_MAX):
        ax.axhline(y, color=_BOUND_COLOR, linewidth=1, linestyle="--", zorder=1)
    if shaded:
        ax.add_patch(Rectangle(
            (ELEC_MIN, GAS_MIN), ELEC_MAX - ELEC_MIN, GAS_MAX - GAS_MIN,
            facecolor="0.85", edgecolor="none", zorder=1.5, label=label,
        ))
    ax.add_patch(Rectangle(
        (ELEC_MIN, GAS_MIN), ELEC_MAX - ELEC_MIN, GAS_MAX - GAS_MIN,
        fill=False, edgecolor="black", linewidth=1.8, zorder=2,
    ))


def _highlight_boundary_ticks(ax, xlim, ylim, x0: float = 0.0):
    x_ticks = sorted(set(ax.get_xticks()) | {x0, ELEC_MIN, ELEC_MAX})
    x_ticks = [t for t in x_ticks if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(x_ticks)
    y_ticks = sorted(set(ax.get_yticks()) | {x0, GAS_MIN, GAS_MAX})
    y_ticks = [t for t in y_ticks if ylim[0] <= t <= ylim[1]]
    ax.set_yticks(y_ticks)
    for lbl, val in zip(ax.get_xticklabels(), x_ticks):
        if val in (ELEC_MIN, ELEC_MAX):
            lbl.set_color(_BOUND_COLOR)
            lbl.set_fontweight("bold")
    for lbl, val in zip(ax.get_yticklabels(), y_ticks):
        if val in (GAS_MIN, GAS_MAX):
            lbl.set_color(_BOUND_COLOR)
            lbl.set_fontweight("bold")


def _cone_polygon(min_ratio: float, max_ratio: float, x_max: float, y_max: float):
    """Vertices of the region {(x, y) in [0, x_max] x [0, y_max] : min_ratio <= y/x <= max_ratio},
    i.e. the convex cone spanned by the two rays, clipped to the plotted box."""
    def exit_point(ratio):
        if ratio * x_max <= y_max:
            return (x_max, ratio * x_max), "right"
        return (y_max / ratio, y_max), "top"

    p_min, edge_min = exit_point(min_ratio)
    p_max, edge_max = exit_point(max_ratio)
    pts = [(0.0, 0.0), p_min]
    if edge_min == "right" and edge_max == "top":
        pts.append((x_max, y_max))
    pts.append(p_max)
    return pts


def plot_angle_ratio_advantage(output_dir: str | None = None, fontsize: int = 11) -> Path:
    base = Path(output_dir) if output_dir else Path("Marius/visualization")
    base.mkdir(parents=True, exist_ok=True)

    lhs_df, _ = create_sample("lhs", N_LHS)

    min_ratio = GAS_MIN / ELEC_MAX  # LR corner
    max_ratio = GAS_MAX / ELEC_MIN  # UL corner
    mid_ratio = ((GAS_MIN + GAS_MAX) / 2) / ((ELEC_MIN + ELEC_MAX) / 2)  # ratio through the rectangle's center

    # ── Left panel extent, origin pinned at (0, 0) ───────────────────────────────
    # the x-extent is a free design choice (padding around the rectangle); the
    # right panel's x-extent is then derived below so both panels end up with
    # identical boxes while keeping x/y on the same scale within each panel
    pad_x = (ELEC_MAX - ELEC_MIN) * LEFT_PAD_FRAC
    xlim_left = (0.0, ELEC_MAX + pad_x)
    ylim_left = (0.0, Y_MAX_LEFT)
    box_aspect = (xlim_left[1] - xlim_left[0]) / (ylim_left[1] - ylim_left[0])

    # ── Right panel extent: y is fixed, x follows from the shared box_aspect ────
    y_max = Y_MAX_RIGHT
    x_max = y_max * box_aspect
    xlim_right = (0.0, x_max)
    ylim_right = (0.0, y_max)

    # equal-width gridspec columns give both panels an identical cell (W, H) --
    # FIG_HEIGHT is chosen so that cell width/height exactly equals box_aspect,
    # so aspect="equal" is a no-op for both (no shrinking, no extending needed)
    avail_frac = GRID_RIGHT - GRID_LEFT
    col_frac = avail_frac / (2 + WSPACE)
    cell_w_in = col_frac * FIG_WIDTH
    cell_h_in = cell_w_in / box_aspect
    fig_height = cell_h_in / (GRID_TOP - GRID_BOTTOM)

    fig = plt.figure(figsize=(FIG_WIDTH, fig_height))
    gs = fig.add_gridspec(1, 2, left=GRID_LEFT, right=GRID_RIGHT, top=GRID_TOP, bottom=GRID_BOTTOM, wspace=WSPACE)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # ── Left: feasible rectangle with the 40 LHS training samples ───────────────
    _draw_rectangle(ax_left, label="Reachable price pairs")
    ax_left.scatter(
        lhs_df["electricity_price"], lhs_df["gas_price"],
        color=_BLUE, s=35, edgecolors="white", linewidths=0.5, zorder=3,
        label=f"{N_LHS} LHS training samples",
    )

    # a single constant-ratio line through the rectangle's center: every point
    # on it has the same c_G/c_el ratio, so the many LHS samples scattered
    # along such lines are redundant for a model that only sees the ratio
    x_line = np.array([0.0, xlim_left[1]])
    y_line = mid_ratio * x_line
    ax_left.plot(x_line, y_line, color=_SLOPE_COLOR, linewidth=1.6, linestyle="-", zorder=3.5)
    rotation_deg = np.degrees(np.arctan(mid_ratio))
    x_elec_mid = (ELEC_MIN + ELEC_MAX) / 2
    ax_left.text(
        x_elec_mid + 0.16 * (ELEC_MAX - ELEC_MIN), mid_ratio * (x_elec_mid + 0.16 * (ELEC_MAX - ELEC_MIN)),
        "same ratio", color=_SLOPE_COLOR, fontsize=fontsize - 1.5, fontweight="bold",
        rotation=rotation_deg, rotation_mode="anchor", ha="left", va="bottom", zorder=3.5,
    )

    ax_left.set_xlim(*xlim_left)
    ax_left.set_ylim(*ylim_left)
    ax_left.set_aspect("equal", adjustable="box")
    _highlight_boundary_ticks(ax_left, xlim_left, ylim_left)
    ax_left.set_xlabel(r"Electricity price $c_{\mathrm{el}}$ [€/MWh]", fontsize=fontsize)
    ax_left.set_ylabel(r"Gas price $c_G$ [€/MWh]", fontsize=fontsize)
    ax_left.set_title("2D LHS sampling", fontsize=fontsize + 1, fontweight="bold")
    ax_left.tick_params(labelsize=fontsize - 1)
    ax_left.grid(True, linewidth=0.4, alpha=0.5)
    ax_left.legend(loc="upper left", fontsize=fontsize - 1.5, framealpha=0.9)

    # ── Right: rectangle + LR/UL rays + shaded convex cone, zoomed way out ──────
    cone_pts = _cone_polygon(min_ratio, max_ratio, x_max, y_max)
    ax_right.add_patch(Polygon(
        cone_pts, closed=True, facecolor=_CONE_FACE, edgecolor="none",
        alpha=0.25, zorder=1.2, label="Reachable price pairs",
    ))

    _draw_rectangle(ax_right)

    # the 38 intermediate angle-ratio training rays -- shown thin and light so the
    # plot doesn't read as if only the two extreme (LR/UL) rays were sampled
    angle_ratios = create_sample("angle", N_LHS).to_numpy()
    for i, ratio in enumerate(angle_ratios[1:-1]):
        end = (x_max, ratio * x_max) if ratio * x_max <= y_max else (y_max / ratio, y_max)
        ax_right.plot(
            [0, end[0]], [0, end[1]], color=_CONE_EDGE, linewidth=0.6, alpha=0.35, zorder=3,
            label="Ratio training samples" if i == 0 else None,
        )

    lr_corner = (ELEC_MAX, GAS_MIN)
    ul_corner = (ELEC_MIN, GAS_MAX)
    lr_end = (x_max, min_ratio * x_max) if min_ratio * x_max <= y_max else (y_max / min_ratio, y_max)
    ul_end = (x_max, max_ratio * x_max) if max_ratio * x_max <= y_max else (y_max / max_ratio, y_max)

    ax_right.plot([0, lr_end[0]], [0, lr_end[1]], color=_CONE_EDGE, linewidth=1.6, zorder=4, label="LR ray (min ratio)")
    ax_right.plot([0, ul_end[0]], [0, ul_end[1]], color=_CONE_EDGE, linewidth=1.6, linestyle="--", zorder=4, label="UL ray (max ratio)")

    ax_right.scatter(*lr_corner, color=_CONE_EDGE, s=45, zorder=5, edgecolors="black", linewidths=0.6)
    ax_right.scatter(*ul_corner, color=_CONE_EDGE, s=45, zorder=5, edgecolors="black", linewidths=0.6)
    ax_right.annotate("LR", lr_corner, textcoords="offset points", xytext=(6, -12), fontsize=fontsize - 1, fontweight="bold", color=_CONE_EDGE)
    ax_right.annotate("UL", ul_corner, textcoords="offset points", xytext=(-18, 6), fontsize=fontsize - 1, fontweight="bold", color=_CONE_EDGE)

    ax_right.scatter([0], [0], color=_BOUND_COLOR, s=35, zorder=5, edgecolors="black", linewidths=0.6)

    ax_right.set_xlim(*xlim_right)
    ax_right.set_ylim(*ylim_right)
    ax_right.set_aspect("equal", adjustable="box")
    _highlight_boundary_ticks(ax_right, xlim_right, ylim_right)
    ax_right.set_xlabel(r"Electricity price $c_{\mathrm{el}}$ [€/MWh]", fontsize=fontsize)
    ax_right.set_ylabel(r"Gas price $c_G$ [€/MWh]", fontsize=fontsize)
    ax_right.set_title("Angle-ratio sampling", fontsize=fontsize + 1, fontweight="bold")
    ax_right.tick_params(labelsize=fontsize - 1)
    ax_right.grid(True, linewidth=0.4, alpha=0.5)
    ax_right.legend(loc="upper left", fontsize=fontsize - 1.5, framealpha=0.9)

    fig.suptitle(
        "Price-space region covered by 2D LHS sampling versus angle-ratio sampling",
        fontsize=fontsize + 3, fontweight="bold", y=0.98,
    )

    # ── Arrow between the two panels ─────────────────────────────────────────────
    # end_margin is kept as an absolute inch distance (not a fraction of the gap)
    # since it exists to clear the right panel's y-axis label/ticks, which have a
    # roughly fixed physical width regardless of how wide the gap is.
    fig.canvas.draw()
    pos_left = ax_left.get_position()
    pos_right = ax_right.get_position()
    y_mid = (pos_left.y0 + pos_left.y1) / 2
    start_margin_in = 0.15
    end_margin_in = 0.75
    x_start = pos_left.x1 + start_margin_in / FIG_WIDTH
    x_end = pos_right.x0 - end_margin_in / FIG_WIDTH

    arrow = FancyArrowPatch(
        (x_start, y_mid), (x_end, y_mid),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=28,
        color="black", linewidth=2.2, zorder=10,
    )
    fig.add_artist(arrow)
    fig.text(
        (x_start + x_end) / 2, y_mid + 0.025,
        r"$r = c_G / c_{\mathrm{el}}$", ha="center", va="bottom",
        fontsize=fontsize, style="italic",
    )

    out_path = base / "angle_ratio_advantage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = plot_angle_ratio_advantage()
    print(f"Saved -> {out}")
