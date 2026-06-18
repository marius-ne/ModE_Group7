"""
Validates the normalized MILP formulation using two kinds of solver calls:

  2-DOF:    solve(r*c_el, c_el) for each (c_el, r) in C_EL_VALS × ratios.
            All lines share the same ratio x-axis.
  1-DOF:    solve(r*C_EL_REF, C_EL_REF, normalize=True, raw_normalized=True)
            → widetilde{OPEX}(r)

Subplot 1 — Normalized view
    OPEX(r·c_el, c_el)/c_el  should collapse onto widetilde{OPEX}(r) for all c_el,
    demonstrating the single-DOF structure.

Subplot 2 — Homogeneity on constant-ratio slopes
    Scatter of solved (c_el, c_G=r·c_el) points coloured by OPEX/c_el.
    On every slope c_G/c_el = r the colour is constant, proving OPEX ∝ c_el.

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
REWRITE    = True
MIP_GAP    = 1e-2
N_CEL      = 6                                  # number of c_el values
N_RATIOS   = 30                                 # shared ratio evaluation points
C_EL_VALS  = np.linspace(0.10, 0.50, N_CEL)    # electricity prices
C_EL_REF   = 0.30                              # reference c_el for 1-DOF solves
ratios     = np.logspace(-1, 1, N_RATIOS)       # r = c_G/c_el in [0.1, 10]
N_SLOPES   = 8                                  # slope lines shown in subplot 2
OUT_PATH   = Path("Marius/visualization/check_MILPs_comparison.png")
CACHE_PATH = Path("Marius/results/check_MILPs_cache.json")

# ---------------------------------------------------------------------------
# Solve or load from cache
# ---------------------------------------------------------------------------
if not REWRITE and CACHE_PATH.exists():
    print(f"Loading cached results from {CACHE_PATH}")
    with open(CACHE_PATH) as fh:
        cache = json.load(fh)
    C_EL_VALS = np.array(cache["C_EL_VALS"])
    C_EL_REF  = float(cache["C_EL_REF"])
    ratios    = np.array(cache["ratios"])
    opex_grid = np.array(cache["opex_grid"])   # (N_CEL, N_RATIOS)
    opex_raw  = np.array(cache["opex_raw"])    # (N_RATIOS,)
else:
    n_total = len(C_EL_VALS) * N_RATIOS + N_RATIOS
    done    = 0

    opex_grid = np.empty((len(C_EL_VALS), N_RATIOS))
    opex_raw  = np.empty(N_RATIOS)

    for i, c_el_i in enumerate(C_EL_VALS):
        for j, r in enumerate(ratios):
            done += 1
            c_G = r * c_el_i
            print(f"[{done:3d}/{n_total}] 2-DOF  c_el={c_el_i:.2f}  r={r:.4f}  c_G={c_G:.4f}")
            opex_grid[i, j] = solve(c_G, c_el_i, mip_gap=MIP_GAP)[0]

    for j, r in enumerate(ratios):
        done += 1
        c_G = r * C_EL_REF
        print(f"[{done:3d}/{n_total}] 1-DOF  c_el={C_EL_REF:.2f}  r={r:.4f}  c_G={c_G:.4f}")
        opex_raw[j] = solve(c_G, C_EL_REF, mip_gap=MIP_GAP,
                            normalize=True, raw_normalized=True)[0]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as fh:
        json.dump({
            "C_EL_VALS": C_EL_VALS.tolist(),
            "C_EL_REF":  C_EL_REF,
            "ratios":    ratios.tolist(),
            "opex_grid": opex_grid.tolist(),
            "opex_raw":  opex_raw.tolist(),
        }, fh, indent=2)
    print(f"Cached to {CACHE_PATH}")

N_CEL     = len(C_EL_VALS)
opex_norm = opex_grid / C_EL_VALS[:, np.newaxis]   # OPEX / c_el; should = opex_raw for all rows

# ---------------------------------------------------------------------------
# Helper: tight inset y-bounds
# ---------------------------------------------------------------------------
def _inset_y_bounds(series, x_lo, x_hi, pad_lo=0.996, pad_hi=1.004):
    """
    Tight (y_lo, y_hi) for an inset on x ∈ [x_lo, x_hi].
    Strict inner indices → ceiling; left-expanded by one neighbor → floor.
    """
    y_for_max, y_for_min = [], []
    for x_arr, y_arr in series:
        x_arr = np.asarray(x_arr, float)
        y_arr = np.asarray(y_arr, float)
        inner = np.where((x_arr >= x_lo) & (x_arr <= x_hi))[0]
        if len(inner) == 0:
            continue
        y_for_max.extend(y_arr[inner].tolist())
        exp_lo = max(0, inner[0] - 1)
        exp_hi = min(len(x_arr) - 1, inner[-1] + 1)
        y_for_min.extend(y_arr[exp_lo:exp_hi + 1].tolist())
    arr_max = np.array([v for v in y_for_max if np.isfinite(v)])
    arr_min = np.array([v for v in y_for_min if np.isfinite(v)])
    return arr_min.min() * pad_lo, arr_max.max() * pad_hi


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
LEGEND_KW   = dict(framealpha=0.85, edgecolor="gray", fontsize=8)
c_el_colors = plt.cm.Blues(np.linspace(0.40, 0.90, N_CEL))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                gridspec_kw={"width_ratios": [1, 1.1]})

# ------------------------------------------------------------------ Subplot 1
def _plot_sp1(ax, with_legend=False):
    ax.fill_between(
        ratios,
        opex_raw * (1 - MIP_GAP),
        opex_raw * (1 + MIP_GAP),
        color="gray", alpha=0.35, zorder=1,
        label=f"±{MIP_GAP:.0%} MIP-gap band" if with_legend else None,
    )
    for i, c_el_i in enumerate(C_EL_VALS):
        ax.plot(ratios, opex_norm[i, :],
                color=c_el_colors[i], marker="o", markersize=3, zorder=3,
                label=f"2-DOF  $c_{{\\rm el}}={c_el_i:.2f}$" if with_legend else None)
    ax.plot(ratios, opex_raw, color="black", marker="s", markersize=3,
            linestyle="--", zorder=4,
            label=r"1-DOF: $\widetilde{\mathrm{OPEX}}(r)$" if with_legend else None)


_plot_sp1(ax1, with_legend=True)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Price ratio $r = c_G / c_{\rm el}$", fontsize=11)
ax1.set_ylabel(r"$\widetilde{\mathrm{OPEX}}$ [kWh]", fontsize=11)
ax1.set_title(
    "Subplot 1 — Normalized view\n"
    r"$\mathrm{OPEX}(r\,c_{\rm el},\,c_{\rm el})\,/\,c_{\rm el}"
    r"= \widetilde{\mathrm{OPEX}}(r)$  for all $c_{\rm el}$",
    fontsize=10,
)
ax1.legend(**LEGEND_KW)
ax1.grid(True, which="both", ls="--", alpha=0.6)

# Inset: zoom r ∈ [0.75, 1.35] — all series share ratios as x-axis
ZOOM_LO, ZOOM_HI = 0.75, 1.35
_sp1_series = [(ratios, opex_norm[i, :]) for i in range(N_CEL)]
_sp1_series.append((ratios, opex_raw))
_ins_y_lo, _ins_y_hi = _inset_y_bounds(_sp1_series, ZOOM_LO, ZOOM_HI)

axins = ax1.inset_axes([0.53, 0.04, 0.44, 0.40])
_plot_sp1(axins, with_legend=False)
axins.set_yscale("log")
axins.set_xlim(ZOOM_LO, ZOOM_HI)
axins.set_ylim(_ins_y_lo, _ins_y_hi)
axins.tick_params(labelsize=7)
axins.set_xlabel(r"$r$", fontsize=8, labelpad=1)
axins.set_ylabel(r"$\widetilde{\mathrm{OPEX}}$", fontsize=8, labelpad=1)
axins.grid(True, which="both", ls="--", alpha=0.5)
axins.set_title(f"Zoom $r\\in[{ZOOM_LO},{ZOOM_HI}]$", fontsize=8, pad=3)
ax1.indicate_inset_zoom(axins, edgecolor="0.4")

# ------------------------------------------------------------------ Subplot 2
# Scatter (c_el, c_G = r·c_el) coloured by OPEX/c_el = widetilde{OPEX}.
# On each slope c_G/c_el = r the colour must be constant (OPEX ∝ c_el).

CEL_pts  = np.repeat(C_EL_VALS, N_RATIOS)
CG_pts   = (C_EL_VALS[:, None] * ratios[None, :]).ravel()
norm_pts = opex_norm.ravel()

log_norm_sp2 = mcolors.LogNorm(vmin=norm_pts.min(), vmax=norm_pts.max())
cmap = plt.cm.plasma.copy()

pcm = ax2.scatter(
    CEL_pts, CG_pts,
    c=norm_pts, norm=log_norm_sp2, cmap=cmap,
    marker="o", edgecolors="white", linewidths=0.5, s=55, zorder=5,
)
fig.colorbar(pcm, ax=ax2, label=r"$\widetilde{\mathrm{OPEX}}$ [kWh]", pad=0.02)

# Constant-ratio slope lines c_G = r · c_el
slope_idx = np.round(np.linspace(0, N_RATIOS - 1, N_SLOPES)).astype(int)
c_el_lo, c_el_hi = C_EL_VALS[0], C_EL_VALS[-1]
for idx in slope_idx:
    r_sel = ratios[idx]
    ax2.plot([c_el_lo, c_el_hi],
             [r_sel * c_el_lo, r_sel * c_el_hi],
             color="white", lw=1.0, alpha=0.8, zorder=3)
    ax2.text(c_el_hi * 1.01, r_sel * c_el_hi,
             f"$r={r_sel:.2f}$", color="white", fontsize=7,
             ha="left", va="center", zorder=4)

ax2.set_xlim(c_el_lo * 0.9, c_el_hi * 1.12)
ax2.set_ylim(ratios[0] * c_el_lo * 0.9,
             ratios[-1] * c_el_hi * 1.05)
ax2.set_xlabel(r"$c_{\rm el}$ [€/kWh]", fontsize=11)
ax2.set_ylabel(r"$c_G$ [€/kWh]", fontsize=11)
ax2.set_title(
    r"Subplot 2 — Homogeneity: colour $= \widetilde{\mathrm{OPEX}} = \mathrm{OPEX}\,/\,c_{\rm el}$"
    "\nConstant colour on each slope "
    r"$\Rightarrow\;\mathrm{OPEX}(r\,c_{\rm el},c_{\rm el})\propto c_{\rm el}$",
    fontsize=10,
)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=200)
plt.close(fig)
print(f"\nFigure saved to {OUT_PATH}")
