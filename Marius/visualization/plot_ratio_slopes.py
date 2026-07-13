"""Feasible c_el/c_G price rectangle with the sampled training ratios (SAMPLING_METHOD:
"log" or "angle") drawn as slopes (rays from the origin) through that same price space."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from pathlib import Path

sys.path.append("Erdem")
from src.sampling.core import GAS_MIN, GAS_MAX, ELEC_MIN, ELEC_MAX, create_sample
from src.visualization.style import apply_style, get_figsize

_BOUND_COLOR = "#888780"

N_RATIOS = 40
PAD_FRAC = 0.6  # fraction of rectangle width/height shown as surrounding margin
SAMPLING_METHOD = "angle"  # "log" -> equally spaced in log(ratio), "angle" -> equally spaced in arctan(ratio)

FIG_WIDTH_CM = 16
CBAR_WIDTH_IN = 1.6  # width the colorbar takes out of the figure, so the axes keep the data aspect


def plot_ratio_slopes(output_dir: str | None = None,
                       sampling_method: str = SAMPLING_METHOD) -> Path:
    base = Path(output_dir) if output_dir else Path("Marius/visualization")
    base.mkdir(parents=True, exist_ok=True)

    ratios = create_sample(sampling_method, N_RATIOS).to_numpy()

    # ── Feasible price rectangle, with padding so the origin is visible ─────────
    pad_x = (ELEC_MAX - ELEC_MIN) * PAD_FRAC
    pad_y = (GAS_MAX - GAS_MIN) * PAD_FRAC
    xlim = (max(0.0, ELEC_MIN - pad_x), ELEC_MAX + pad_x)
    ylim = (max(0.0, GAS_MIN - pad_y), GAS_MAX + pad_y)

    # the figure width comes from the style; its height is then whatever makes the axes match
    # the data's true aspect ratio, so the equal-aspect axes below don't leave the box shrunk
    # with excess whitespace
    apply_style(width_cm=FIG_WIDTH_CM, grid=True, strict=True)
    fig_width, _ = get_figsize(width_cm=FIG_WIDTH_CM)
    fig_height = (fig_width - CBAR_WIDTH_IN) * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")  # 1 unit of c_el == 1 unit of c_G, so slopes/angles are true to scale

    for x in (ELEC_MIN, ELEC_MAX):
        ax.axvline(x, color=_BOUND_COLOR, linewidth=1, linestyle="--", zorder=1)
    for y in (GAS_MIN, GAS_MAX):
        ax.axhline(y, color=_BOUND_COLOR, linewidth=1, linestyle="--", zorder=1)

    # shade the actual feasible rectangle so it reads as distinct from the dashed
    # lines that continue across the full plot
    ax.add_patch(Rectangle(
        (ELEC_MIN, GAS_MIN), ELEC_MAX - ELEC_MIN, GAS_MAX - GAS_MIN,
        facecolor="0.85", edgecolor="none", zorder=1.5,
    ))

    # ── Ratios as rays from the origin, colored sequentially (light -> dark) by
    # log(ratio) since the ratios themselves are log-spaced ─────────────────────
    cmap = plt.get_cmap("viridis")
    log_ratios = np.log10(ratios)
    norm = plt.Normalize(vmin=log_ratios.min(), vmax=log_ratios.max())

    # crisp black outline of the actual rectangle
    ax.add_patch(Rectangle(
        (ELEC_MIN, GAS_MIN), ELEC_MAX - ELEC_MIN, GAS_MAX - GAS_MIN,
        fill=False, edgecolor="black", linewidth=1.8, zorder=2,
    ))

    for ratio, log_r in zip(ratios, log_ratios):
        x_end = min(xlim[1], ylim[1] / ratio)
        ax.plot([0, x_end], [0, ratio * x_end], color=cmap(norm(log_r)),
                 linewidth=1.1, alpha=0.75, zorder=4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    tick_ratios = np.array([ratios.min(), 0.1, 0.3, 1.0, 3.0, ratios.max()])
    tick_ratios = tick_ratios[(tick_ratios >= ratios.min()) & (tick_ratios <= ratios.max())]
    cbar.set_ticks(np.log10(tick_ratios))
    cbar.set_ticklabels([f"{r:.3g}" for r in tick_ratios])
    cbar.set_label(r"Price ratio $c_G\,/\,c_{\mathrm{el}}$ [-]")

    # origin marker, drawn on top so it stays legible
    ax.scatter([0], [0], color=_BOUND_COLOR, s=35, zorder=5, edgecolors="black", linewidths=0.6)

    # add the boundary values as extra, highlighted ticks where the lines meet the axes
    x_ticks = sorted(set(ax.get_xticks()) | {0.0, ELEC_MIN, ELEC_MAX})
    x_ticks = [t for t in x_ticks if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(x_ticks)
    y_ticks = sorted(set(ax.get_yticks()) | {0.0, GAS_MIN, GAS_MAX})
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

    ax.set_xlabel(r"Electricity price $c_{\mathrm{el}}$ [€/MWh]")
    ax.set_ylabel(r"Gas price $c_G$ [€/MWh]")
    ax.set_title(
        f"Feasible price rectangle with {N_RATIOS} {sampling_method}-sampled ratios as slopes"
    )

    out_path = base / f"ratio_slopes_{sampling_method}.png"
    fig.savefig(out_path)  # dpi/bbox come from apply_style's rcParams
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    for method in ("log", "angle"):
        out = plot_ratio_slopes(sampling_method=method)
        print(f"Saved -> {out}")
