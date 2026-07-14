"""Two-panel illustration of why angle-ratio sampling extrapolates further than
plain 2D LHS sampling. Left: the feasible c_el/c_G price rectangle with the 40
LHS training samples (as in Erdem's sampling). Right: the same rectangle with
only its LR (min-ratio) and UL (max-ratio) corner rays drawn from the origin,
and the convex cone spanned by those rays shaded -- illustrating that a model
trained on the ratio alone generalizes to any price pair on that cone, a far
larger region than the original rectangle."""

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, Arc
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.patheffects import withStroke
from pathlib import Path

sys.path.append("Erdem")
from src.sampling.core import GAS_MIN, GAS_MAX, ELEC_MIN, ELEC_MAX, create_sample
from src.visualization.style import apply_style, get_figsize

_BOUND_COLOR = "#888780"
_BLUE = "#2166AC"
_CONE_FACE = "#D6604D"
_CONE_EDGE = "#B2182B"
_SLOPE_COLOR = "#B2182B"
_THETA_COLOR = "#E08214"

N_LHS = 40
LEFT_PAD_FRAC = 0.25    # padding around the rectangle in the left panel's x-axis
Y_MAX_LEFT = 430.0       # left panel's y-axis extent
Y_MAX_RIGHT = 640.0      # right panel's y-axis extent

FIG_WIDTH_CM = 22
# margins in figure fractions: the top leaves a little room above the panel titles, the
# wspace for the arrow (and its label) between the panels plus the right panel's y-axis labels
GRID_LEFT, GRID_RIGHT, GRID_TOP, GRID_BOTTOM, WSPACE = 0.07, 0.97, 0.92, 0.19, 0.45


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


def _merge_ticks(auto_ticks, special_ticks, lo: float, hi: float, min_sep_frac: float = 0.05):
    """Auto ticks unioned with the special (boundary) ticks, dropping any auto tick that falls
    within min_sep_frac of the axis span from a special one. Without this, a boundary value
    close to a round auto tick (e.g. GAS_MAX=315 next to an auto tick at 300) produces two
    labels close enough to overlap."""
    min_sep = min_sep_frac * (hi - lo)
    kept_auto = [t for t in auto_ticks if all(abs(t - s) >= min_sep for s in special_ticks)]
    ticks = sorted(set(kept_auto) | set(special_ticks))
    return [t for t in ticks if lo <= t <= hi]


def _highlight_boundary_ticks(ax, xlim, ylim, x0: float = 0.0):
    x_auto = MaxNLocator(nbins=5).tick_values(*xlim)
    x_ticks = _merge_ticks(x_auto, {x0, ELEC_MIN, ELEC_MAX}, *xlim)
    ax.set_xticks(x_ticks)
    y_auto = MaxNLocator(nbins=5).tick_values(*ylim)
    y_ticks = _merge_ticks(y_auto, {x0, GAS_MIN, GAS_MAX}, *ylim)
    ax.set_yticks(y_ticks)
    # the irregular special+auto tick spacing confuses the default AutoMinorLocator into
    # inserting a variable, cluttered number of minor ticks -- pin it to a fixed count
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
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


def plot_angle_ratio_advantage(output_dir: str | None = None) -> Path:
    base = Path(output_dir) if output_dir else Path("Marius/visualization")
    base.mkdir(parents=True, exist_ok=True)

    # the figure's geometry is worked out by hand below, so only the width is taken from the
    # style; the in-axes annotations are sized relative to its base font size
    apply_style(width_cm=FIG_WIDTH_CM, grid=True, strict=True)
    fig_width, _ = get_figsize(width_cm=FIG_WIDTH_CM)
    fontsize = mpl.rcParams["font.size"]

    lhs_df, _ = create_sample("lhs", N_LHS)

    min_ratio = GAS_MIN / ELEC_MAX  # LR corner
    max_ratio = GAS_MAX / ELEC_MIN  # UL corner
    mid_ratio = 1.01*((GAS_MIN + GAS_MAX) / 2) / ((ELEC_MIN + ELEC_MAX) / 2)  # ratio through the rectangle's center

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
    cell_w_in = col_frac * fig_width
    cell_h_in = cell_w_in / box_aspect
    fig_height = cell_h_in / (GRID_TOP - GRID_BOTTOM)

    fig = plt.figure(figsize=(fig_width, fig_height))
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
        rotation=rotation_deg, rotation_mode="anchor", ha="left", va="top", zorder=3.5,
    )

    ax_left.set_xlim(*xlim_left)
    ax_left.set_ylim(*ylim_left)
    ax_left.set_aspect("equal", adjustable="box")
    _highlight_boundary_ticks(ax_left, xlim_left, ylim_left)
    ax_left.set_xlabel(r"Electricity price $c_{\mathrm{el}}$ [€/MWh]")
    ax_left.set_ylabel(r"Gas price $c_{\mathrm{gas}}$ [€/MWh]")
    ax_left.legend(loc="upper left")

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
            label=f"{N_LHS} ratio training samples" if i == 0 else None,
        )

    lr_corner = (ELEC_MAX, GAS_MIN)
    ul_corner = (ELEC_MIN, GAS_MAX)
    lr_end = (x_max, min_ratio * x_max) if min_ratio * x_max <= y_max else (y_max / min_ratio, y_max)
    ul_end = (x_max, max_ratio * x_max) if max_ratio * x_max <= y_max else (y_max / max_ratio, y_max)

    ax_right.plot([0, lr_end[0]], [0, lr_end[1]], color=_CONE_EDGE, linewidth=1.6, zorder=4, label="LR ray (min ratio)")
    ax_right.plot([0, ul_end[0]], [0, ul_end[1]], color=_CONE_EDGE, linewidth=1.6, linestyle="--", zorder=4, label="UL ray (max ratio)")

    ax_right.scatter(*lr_corner, color=_CONE_EDGE, s=45, zorder=5, edgecolors="black", linewidths=0.6)
    ax_right.scatter(*ul_corner, color=_CONE_EDGE, s=45, zorder=5, edgecolors="black", linewidths=0.6)
    ax_right.annotate("LR", lr_corner, textcoords="offset points", xytext=(6, 8), fontsize=fontsize - 1, fontweight="bold", color=_CONE_EDGE)
    ax_right.annotate("UL", ul_corner, textcoords="offset points", xytext=(-18, 6), fontsize=fontsize - 1, fontweight="bold", color=_CONE_EDGE)

    # theta_min: the LR ray's angle to the x-axis, drawn as an arc in the empty
    # wedge below the ray so it reads as an angle marker without cluttering the cone
    theta_min_deg = np.degrees(np.arctan(min_ratio))
    arc_r = 0.85 * x_max
    ax_right.add_patch(Arc(
        (0, 0), 2 * arc_r, 2 * arc_r, angle=0, theta1=0, theta2=theta_min_deg,
        color=_THETA_COLOR, linewidth=1.2, capstyle="butt", zorder=4,
    ))
    # label sits below the x-axis, out of the cone/ray clutter entirely, with a thin
    # leader line back up to the arc -- same device plot_pareto_accuracy_vs_time.py
    # uses for its LP-mean point, whose label would otherwise land on the x-axis
    theta_mid_rad = np.radians(theta_min_deg / 2)
    theta_tip = (arc_r * np.cos(theta_mid_rad), arc_r * np.sin(theta_mid_rad))
    theta_label = ax_right.annotate(
        r"$\theta_{\min}$", theta_tip, textcoords="offset points", xytext=(12, -15),
        color=_THETA_COLOR, fontsize=fontsize - 1, ha="center", va="top", zorder=4,
        arrowprops=dict(arrowstyle="-", color=_THETA_COLOR, linewidth=0.8, shrinkA=2, shrinkB=0),
        annotation_clip=False,
    )
    # the "cm" mathtext fontset has no bold glyphs for \mathbf/\boldsymbol, so fake
    # bold with a same-color stroke outline instead
    theta_label.set_path_effects([withStroke(linewidth=0.7, foreground=_THETA_COLOR)])
    theta_label.arrow_patch.set_clip_on(False)

    ax_right.scatter([0], [0], color=_BOUND_COLOR, s=35, zorder=5, edgecolors="black", linewidths=0.6)

    ax_right.set_xlim(*xlim_right)
    ax_right.set_ylim(*ylim_right)
    ax_right.set_aspect("equal", adjustable="box")
    _highlight_boundary_ticks(ax_right, xlim_right, ylim_right)
    ax_right.set_xlabel(r"Electricity price $c_{\mathrm{el}}$ [€/MWh]")
    ax_right.set_ylabel(r"Gas price $c_{\mathrm{gas}}$ [€/MWh]")
    ax_right.legend(loc="upper left")

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
    x_start = pos_left.x1 + start_margin_in / fig_width
    x_end = pos_right.x0 - end_margin_in / fig_width

    arrow = FancyArrowPatch(
        (x_start, y_mid), (x_end, y_mid),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=28,
        color="black", linewidth=2.2, zorder=10,
    )
    fig.add_artist(arrow)
    fig.text(
        (x_start + x_end) / 2, y_mid + 0.05,
        r"$r = c_{\mathrm{gas}} / c_{\mathrm{el}}$", ha="center", va="bottom",
        fontsize=fontsize, style="italic",
    )

    out_path = base / "angle_ratio_advantage.png"
    fig.savefig(out_path)  # dpi/bbox come from apply_style's rcParams
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = plot_angle_ratio_advantage()
    print(f"Saved -> {out}")
