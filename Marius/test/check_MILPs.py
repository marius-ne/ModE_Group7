"""
Validates the normalized MILP formulation using two kinds of solver calls:

  2-DOF grid:  solve(r*c_el, c_el)  for each (c_el, r) in C_EL_GRID x ratios
  1-DOF norm:  solve(r*C_EL, C_EL, normalize=True, raw_normalized=True)
               only at the single reference price C_EL  ->  OPEX / C_EL

Subplot 1 — Normalized view (ratio on x)
    OPEX(r*c_el, c_el) / c_el  for every c_el in C_EL_GRID should collapse
    onto the single 1-DOF curve OPEX_raw(r), demonstrating that the problem
    has only one degree of freedom.

Subplot 2 — Original view (c_el x c_G plane)
    Heatmap: OPEX reconstructed analytically as c_el * OPEX_raw(c_G/c_el).
    Circles  (white edge): 2-DOF grid solves — multiple points per slope line.
    Diamonds (red edge):   1-DOF normalized solves at c_el = C_EL.
    If both match the heatmap colour at their location the equivalence holds.

Usage:
    python Marius/test/check_MILPs.py
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, "Marius")
from formulation_MILP import solve

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REWRITE  = False
MIP_GAP  = 1e-2
N_RATIOS    = 30
C_EL        = 0.30                              # reference price for 1-DOF solves
C_EL_GRID   = np.array([0.10, 0.20, 0.30, 0.50])  # c_el values for 2-DOF grid
OUT_PATH    = Path("Marius/visualization/check_MILPs_comparison.png")
CACHE_PATH  = Path("Marius/results/check_MILPs_cache.json")

ratios = np.logspace(-1, 1, N_RATIOS)   # r = c_G / c_el  in [0.1, 10]

# ---------------------------------------------------------------------------
# Solve or load from cache
# ---------------------------------------------------------------------------
if not REWRITE and CACHE_PATH.exists():
    print(f"Loading cached results from {CACHE_PATH}")
    with open(CACHE_PATH) as fh:
        cache = json.load(fh)
    ratios         = np.array(cache["ratios"])
    C_EL_GRID      = np.array(cache["C_EL_GRID"])
    C_EL           = float(cache["C_EL"])
    opex_2dof_grid = np.array(cache["opex_2dof_grid"])  # (len(C_EL_GRID), N_RATIOS)
    opex_raw       = np.array(cache["opex_raw"])         # (N_RATIOS,)
else:
    n_total = len(C_EL_GRID) * N_RATIOS + N_RATIOS
    done = 0

    opex_2dof_grid = np.empty((len(C_EL_GRID), N_RATIOS))
    opex_raw       = np.empty(N_RATIOS)

    # 2-DOF grid
    for i, c_el_i in enumerate(C_EL_GRID):
        for j, r in enumerate(ratios):
            done += 1
            c_G = r * c_el_i
            print(f"[{done:3d}/{n_total}] 2-DOF  c_el={c_el_i:.2f}  r={r:.4f}  c_G={c_G:.4f}")
            opex_2dof_grid[i, j] = solve(c_G, c_el_i, mip_gap=MIP_GAP)[0]

    # 1-DOF normalized — only at C_EL
    for j, r in enumerate(ratios):
        done += 1
        c_G = r * C_EL
        print(f"[{done:3d}/{n_total}] 1-DOF  c_el={C_EL:.2f}  r={r:.4f}  c_G={c_G:.4f}")
        opex_raw[j] = solve(c_G, C_EL, mip_gap=MIP_GAP,
                            normalize=True, raw_normalized=True)[0]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as fh:
        json.dump({
            "C_EL": C_EL,
            "C_EL_GRID": C_EL_GRID.tolist(),
            "ratios": ratios.tolist(),
            "opex_2dof_grid": opex_2dof_grid.tolist(),
            "opex_raw": opex_raw.tolist(),
        }, fh, indent=2)
    print(f"Cached to {CACHE_PATH}")

# ---------------------------------------------------------------------------
# Build heatmap analytically (no extra solver calls)
# ---------------------------------------------------------------------------
N_GRID   = 300
c_el_arr = np.linspace(0.05, 0.60, N_GRID)   # linear x-axis
c_G_arr  = np.linspace(0.01, 5.00, N_GRID)   # linear y-axis
CEL_mesh, CG_mesh = np.meshgrid(c_el_arr, c_G_arr)
ratio_mesh = CG_mesh / CEL_mesh

log_r_sweep  = np.log(ratios)
log_raw_vals = np.log(opex_raw)
log_r_clipped = np.log(np.clip(ratio_mesh, ratios[0], ratios[-1]))
OPEX_mesh = CEL_mesh * np.exp(np.interp(log_r_clipped, log_r_sweep, log_raw_vals))
OPEX_mesh = np.where(
    (ratio_mesh >= ratios[0]) & (ratio_mesh <= ratios[-1]),
    OPEX_mesh, np.nan,
)

vmin = np.nanmin(OPEX_mesh)
vmax = np.nanmax(OPEX_mesh)
log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
LEGEND_KW = dict(framealpha=0.85, edgecolor="gray", fontsize=8)
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 6),
    gridspec_kw={"width_ratios": [1, 1.45]},
)

# ------------------------------------------------------------------ Subplot 1
# Show OPEX/c_el vs ratio for every c_el in C_EL_GRID — they should
# collapse onto the single 1-DOF reference curve opex_raw.

c_el_colors = plt.cm.Blues(np.linspace(0.45, 0.90, len(C_EL_GRID)))

def _plot_sp1(ax, ratios, opex_2dof_grid, opex_raw, C_EL_GRID, c_el_colors,
              MIP_GAP, with_legend=False):
    """Draw the subplot-1 content on *ax* (used for main axes and inset)."""
    ax.fill_between(
        ratios,
        opex_raw * (1 - MIP_GAP),
        opex_raw * (1 + MIP_GAP),
        color="gray", alpha=0.35, zorder=1,
        label=f"±{MIP_GAP:.0%} MIP-gap band",
    )
    for i, c_el_i in enumerate(C_EL_GRID):
        ax.plot(ratios, opex_2dof_grid[i, :] / c_el_i,
                color=c_el_colors[i], marker="o", markersize=3, zorder=3,
                label=f"2-DOF / $c_{{\\rm el}}$  ($c_{{\\rm el}}={c_el_i}$)" if with_legend else None)
    ax.plot(ratios, opex_raw, color="black", marker="s", markersize=3,
            linestyle="--", zorder=4,
            label=r"1-DOF: $\mathrm{OPEX}_{\rm raw}(r)$" if with_legend else None)

_plot_sp1(ax1, ratios, opex_2dof_grid, opex_raw, C_EL_GRID, c_el_colors,
          MIP_GAP, with_legend=True)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Price ratio $r = c_G / c_{\rm el}$", fontsize=11)
ax1.set_ylabel(r"OPEX / $c_{\rm el}$  [kWh]", fontsize=11)
ax1.set_title(
    "Subplot 1 — Normalized view\n"
    r"2-DOF lines $\mathrm{OPEX}(r \cdot c_{\rm el},\,c_{\rm el})/c_{\rm el}$"
    " collapse onto 1-DOF",
    fontsize=10,
)
ax1.legend(**LEGEND_KW)
ax1.grid(True, which="both", ls="--", alpha=0.6)

# Inset: zoom into a narrow x-window around r≈1 so the ±1 % band is legible.
ZOOM_LO, ZOOM_HI = 0.75, 1.35
# Center y-limits on opex_raw at r≈1 with a fixed half-width of 1.5×MIP_GAP
# so the ±1% band fills ~67% of the inset height regardless of data spread.
idx_center = np.argmin(np.abs(ratios - 1.0))
opex_center = opex_raw[idx_center]
ZOOM_Y_HALF = 1.5 * MIP_GAP
y_lo = opex_center * (1 - ZOOM_Y_HALF)
y_hi = opex_center * (1 + ZOOM_Y_HALF)

axins = ax1.inset_axes([0.53, 0.04, 0.44, 0.40])   # lower-right corner
_plot_sp1(axins, ratios, opex_2dof_grid, opex_raw, C_EL_GRID, c_el_colors,
          MIP_GAP, with_legend=False)
axins.set_yscale("log")
axins.set_xlim(ZOOM_LO, ZOOM_HI)
axins.set_ylim(y_lo, y_hi)
axins.tick_params(labelsize=7)
axins.set_xlabel(r"$r$", fontsize=8, labelpad=1)
axins.set_ylabel(r"OPEX/$c_{\rm el}$", fontsize=8, labelpad=1)
axins.grid(True, which="both", ls="--", alpha=0.5)
axins.set_title(f"Zoom r∈[{ZOOM_LO},{ZOOM_HI}]", fontsize=8, pad=3)
ax1.indicate_inset_zoom(axins, edgecolor="0.4")

# ------------------------------------------------------------------ Subplot 2
cmap = plt.cm.plasma.copy()
cmap.set_bad(color="#dddddd")

pcm = ax2.pcolormesh(
    CEL_mesh, CG_mesh, OPEX_mesh,
    norm=log_norm, cmap=cmap, shading="auto",
)
fig.colorbar(pcm, ax=ax2, label="OPEX [€]", pad=0.02)

# Constant-ratio slope lines (linear axes → straight lines)
n_lines = 8
slope_idx = np.round(np.linspace(0, N_RATIOS - 1, n_lines)).astype(int)
for idx in slope_idx:
    r_sel = ratios[idx]
    c_G_line = r_sel * c_el_arr
    in_domain = (c_G_line >= c_G_arr[0]) & (c_G_line <= c_G_arr[-1])
    if in_domain.any():
        ax2.plot(c_el_arr[in_domain], c_G_line[in_domain],
                 color="white", lw=0.9, alpha=0.75, zorder=3)
        ax2.text(c_el_arr[in_domain][-1] * 0.99, c_G_line[in_domain][-1],
                 f"r={r_sel:.2f}", color="white", fontsize=7.5,
                 ha="right", va="center", zorder=4)

# 2-DOF grid — circles with white edge
for i, c_el_i in enumerate(C_EL_GRID):
    c_G_vals = ratios * c_el_i
    in_view = (c_G_vals >= c_G_arr[0]) & (c_G_vals <= c_G_arr[-1])
    ax2.scatter(
        np.full(in_view.sum(), c_el_i), c_G_vals[in_view],
        c=opex_2dof_grid[i, in_view], norm=log_norm, cmap=cmap,
        marker="o", edgecolors="white", linewidths=0.7, s=45, zorder=5,
    )

# 1-DOF normalized — diamonds with red edge (visually distinct)
solved_c_G = ratios * C_EL
in_norm = (solved_c_G >= c_G_arr[0]) & (solved_c_G <= c_G_arr[-1])
ax2.axvline(C_EL, color="white", lw=1.2, ls="--", alpha=0.8, zorder=4)
ax2.scatter(
    np.full(in_norm.sum(), C_EL), solved_c_G[in_norm],
    c=(opex_raw * C_EL)[in_norm], norm=log_norm, cmap=cmap,
    marker="D", edgecolors="red", linewidths=1.1, s=60, zorder=6,
)

# Legend proxy artists
ax2.scatter([], [], marker="o", edgecolors="white", linewidths=0.7, s=45,
            color="gray", label="2-DOF grid (circles)")
ax2.scatter([], [], marker="D", edgecolors="red", linewidths=1.1, s=60,
            color="gray", label=f"1-DOF normalized, $c_{{\\rm el}}={C_EL}$ (diamonds)")

ax2.set_xlabel(r"$c_{\rm el}$ [€/kWh]", fontsize=11)
ax2.set_ylabel(r"$c_G$ [€/kWh]", fontsize=11)
ax2.set_title(
    "Subplot 2 — Original view\n"
    "Heatmap from 1-DOF; dots coloured by actual OPEX — match = equivalence",
    fontsize=10,
)
ax2.legend(**LEGEND_KW)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=200)
plt.close(fig)
print(f"\nFigure saved to {OUT_PATH}")
