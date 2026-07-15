"""Grid over the feasible c_el/c_G price rectangle, and the resulting price-ratio distribution."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("Erdem")
from src.sampling.core import GAS_MIN, GAS_MAX, ELEC_MIN, ELEC_MAX
from src.visualization.style import apply_style

_BLUE = "#2166AC"
_TEAL = "#35978F"
_BOUND_COLOR = "#888780"

N_GRID = 50
PAD_FRAC = 0.6  # fraction of rectangle width/height shown as surrounding margin


def plot_price_grid_ratio(output_dir: str | None = None) -> Path:
    base = Path(output_dir) if output_dir else Path("Marius/visualization")
    base.mkdir(parents=True, exist_ok=True)

    c_el_vals = np.linspace(ELEC_MIN, ELEC_MAX, N_GRID)
    c_G_vals = np.linspace(GAS_MIN, GAS_MAX, N_GRID)
    c_el_grid, c_G_grid = np.meshgrid(c_el_vals, c_G_vals)
    ratios = (c_G_grid / c_el_grid).ravel()

    apply_style(width_cm=22, aspect=2.6, nrows=2, grid=True, strict=True)
    fig, ((ax_grid, ax_hist), (ax_empty, ax_hist_log)) = plt.subplots(
        2, 2, constrained_layout=True
    )
    ax_empty.axis("off")

    # ── Left: feasible price rectangle with 20x20 grid, zoomed out ──────────────
    for x in (ELEC_MIN, ELEC_MAX):
        ax_grid.axvline(x, color=_BOUND_COLOR, linewidth=1, linestyle="--", zorder=1)
    for y in (GAS_MIN, GAS_MAX):
        ax_grid.axhline(y, color=_BOUND_COLOR, linewidth=1, linestyle="--", zorder=1)
    ax_grid.scatter(c_el_grid.ravel(), c_G_grid.ravel(), color=_BLUE, s=3,
                     edgecolors="white", linewidths=0.2, zorder=2)

    pad_x = (ELEC_MAX - ELEC_MIN) * PAD_FRAC
    pad_y = (GAS_MAX - GAS_MIN) * PAD_FRAC
    xlim = (max(0.0, ELEC_MIN - pad_x), ELEC_MAX + pad_x)
    ylim = (max(0.0, GAS_MIN - pad_y), GAS_MAX + pad_y)
    ax_grid.set_xlim(*xlim)
    ax_grid.set_ylim(*ylim)

    # add the boundary values as extra, highlighted ticks where the lines meet the axes
    x_ticks = sorted(set(ax_grid.get_xticks()) | {ELEC_MIN, ELEC_MAX})
    x_ticks = [t for t in x_ticks if xlim[0] <= t <= xlim[1]]
    ax_grid.set_xticks(x_ticks)
    y_ticks = sorted(set(ax_grid.get_yticks()) | {GAS_MIN, GAS_MAX})
    y_ticks = [t for t in y_ticks if ylim[0] <= t <= ylim[1]]
    ax_grid.set_yticks(y_ticks)
    for lbl, val in zip(ax_grid.get_xticklabels(), x_ticks):
        if val in (ELEC_MIN, ELEC_MAX):
            lbl.set_color(_BOUND_COLOR)
            lbl.set_fontweight("bold")
    for lbl, val in zip(ax_grid.get_yticklabels(), y_ticks):
        if val in (GAS_MIN, GAS_MAX):
            lbl.set_color(_BOUND_COLOR)
            lbl.set_fontweight("bold")

    ax_grid.set_xlabel(r"Electricity price $c_{\mathrm{el}}$ [€/MWh]")
    ax_grid.set_ylabel(r"Gas price $c_G$ [€/MWh]")
    ax_grid.set_title(f"Feasible price rectangle — {N_GRID}×{N_GRID} grid")

    # ── Right: PDF histogram of resulting price ratios ──────────────────────────
    ax_hist.hist(ratios, bins=20, density=True, color=_TEAL, edgecolor="white", linewidth=0.5)
    ax_hist.set_xlabel(r"Price ratio $c_G\,/\,c_{\mathrm{el}}$ [-]")
    ax_hist.set_ylabel("Probability density")
    ax_hist.set_title(f"Price ratios of the {len(ratios)} grid samples")

    # ── Bottom-right: same histogram with a log x-axis ──────────────────────────
    log_bins = np.logspace(np.log10(ratios.min()), np.log10(ratios.max()), 21)
    ax_hist_log.hist(ratios, bins=log_bins, density=True, color=_TEAL, edgecolor="white", linewidth=0.5)
    ax_hist_log.set_xscale("log")
    ax_hist_log.set_xlabel(r"Price ratio $c_G\,/\,c_{\mathrm{el}}$ [-]")
    ax_hist_log.set_ylabel("Probability density")
    ax_hist_log.set_title(f"Price ratios of the {len(ratios)} grid samples — log scale")

    out_path = base / "price_grid_ratio.png"
    fig.savefig(out_path)  # dpi/bbox come from apply_style's rcParams
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = plot_price_grid_ratio()
    print(f"Saved → {out}")
