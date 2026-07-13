"""OPEX vs. price ratio for all four formulations, and how tightly the LP bounds bracket the
MILP.

MILP, LP lower, LP upper (mode="min") and LP approx (mode="mean_efficiency") are read straight
from the 1D_angle training set that Marius/surrogate_models/run_full_pipeline.py already
solved (the 40 angle-sampled ratios the 1D_angle surrogate is fitted on), not re-solved here.
In 1D mode that training set's opex_* columns are specific OPEX (OPEX / c_el, with c_el pinned
to _evaluation_common.C_EL_REF = 1.0), which is what the y-axis already shows.

LP upper's two heuristics (boilers_on, chp_on) are the one thing the training set does not
carry -- it only keeps the cheaper of the two (mode="min") -- so those are solved here
separately, on the same ratios, and cached to opex_lp_upper_modes_angle_40.csv.

Two separate figures, each its own file:
  opex_vs_price_ratio_linear.png/.pdf   linear axes.
  opex_vs_price_ratio_log.png/.pdf      log-log axes, with the mean/worst LP-bound gap noted.
Each carries two insets: a low-r zoom (leftmost decade tenth) and a zoom on the r in [0.8, 1.1]
transition region around the CHP/boiler break-even -- sized to fully frame the four main
curves (MILP, LP lower, LP upper min, LP approx) within that x-range.

    python Marius/visualization/plot_opex_over_ratio.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))
sys.path.insert(0, str(ROOT / "Marius" / "evaluation"))

from _evaluation_common import C_EL_REF, P_D, Q_D, STRICT_DEMAND_SATISFACTION
from src.optimization.core import solve_lp_upper
from src.visualization.style import apply_style

OUT_DIR = ROOT / "Marius" / "visualization"
RESULTS_DIR = ROOT / "Marius" / "results"

# The 1D_angle training set: the 40 angle-sampled ratios the 1D_angle surrogate is fitted on,
# already solved for MILP/LP lower/LP upper(min)/LP approx by run_full_pipeline.py.
RATIO_SOURCE_CSV = RESULTS_DIR / "evaluation_40_training_samples_1D_angle.csv"
# LP upper's boilers_on/chp_on breakdown, solved and cached by this script (see _lp_upper_modes).
LP_UPPER_MODES_CSV = RESULTS_DIR / "opex_lp_upper_modes_angle_40.csv"
REUSE_EXISTING_MODES = True

if not RATIO_SOURCE_CSV.exists():
    raise FileNotFoundError(
        f"{RATIO_SOURCE_CSV} not found. It is the 1D_angle training set produced by "
        f"Marius/surrogate_models/run_full_pipeline.py (N_TRAIN=40)."
    )
_df = pd.read_csv(RATIO_SOURCE_CSV)
print(f"Read {len(_df)} solved training points from {RATIO_SOURCE_CSV}")

price_ratios = _df["ratio"].to_numpy(dtype=float)
opex_milp_values = _df["opex_milp"].to_numpy(dtype=float)
opex_lp_lower_values = _df["opex_lp_lower"].to_numpy(dtype=float)
opex_lp_upper_values = _df["opex_lp_upper"].to_numpy(dtype=float)  # mode="min"
opex_lp_approx_mean_values = _df["opex_lp_approx"].to_numpy(dtype=float)


def _lp_upper_modes() -> pd.DataFrame:
    """LP upper's boilers_on/chp_on solved separately, for the faint context lines around the
    mode="min" curve -- reused from disk if this exact ratio sweep is already solved."""
    if REUSE_EXISTING_MODES and LP_UPPER_MODES_CSV.exists():
        cached = pd.read_csv(LP_UPPER_MODES_CSV)
        if (len(cached) == len(price_ratios)
                and np.allclose(cached["ratio"].to_numpy(), price_ratios)):
            print(f"Reusing LP upper mode breakdown from {LP_UPPER_MODES_CSV}")
            return cached
        print(f"Cached {LP_UPPER_MODES_CSV} is for a different ratio sweep -- re-solving.")

    print(f"Solving LP upper's boilers_on/chp_on modes for {len(price_ratios)} ratios ...")
    rows = []
    for r in price_ratios:
        c_g = r * C_EL_REF
        (opex_bo, _), (opex_chp, _) = solve_lp_upper(
            Q_D, P_D, c_g, C_EL_REF, return_both=True,
            strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION,
        )
        rows.append({"ratio": r, "opex_lp_upper_bo": opex_bo, "opex_lp_upper_chp": opex_chp})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LP_UPPER_MODES_CSV, index=False)
    print(f"Saved LP upper mode breakdown to {LP_UPPER_MODES_CSV}")
    return df


_modes_df = _lp_upper_modes()
opex_lp_upper_bo_values = _modes_df["opex_lp_upper_bo"].to_numpy(dtype=float)
opex_lp_upper_chp_values = _modes_df["opex_lp_upper_chp"].to_numpy(dtype=float)

OPEX_YLABEL = r"$\widetilde{\mathrm{OPEX}}$ $\left[\dfrac{€_{\mathrm{OPEX}}}{€\,/\,\mathrm{kWh}}\right]$"
MS = 2.5   # uniform marker size

# Palette (ColorBrewer-inspired, perceptually distinct)
C_MILP        = "#2166AC"   # strong blue
C_LP_LOWER    = "#4DAC26"   # medium green
C_LP_UB_BO    = "#F4A582"   # light salmon  (faint boilers_on)
C_LP_UB_CHP   = "#FDCC8A"   # light amber   (faint chp_on)
C_LP_UB_MIN   = "#D6604D"   # muted red     (best upper)
C_LP_APPROX   = "#35978F"   # teal

# Sanity checks: bounds must bracket the MILP solution at every point.
# The MILP is solved with mip_gap=1e-2, so its returned objective can be up to 1% above
# the true optimum. LP upper is also a feasible MILP solution, so LP_upper < MILP_returned
# is normal within that gap. Only flag if LP_upper < MILP_returned * (1 - MIP_GAP),
# which would place LP_upper below the solver's own lower bound — a genuine error.
_MIP_GAP = 1e-2
_fin = np.isfinite(opex_milp_values) & np.isfinite(opex_lp_lower_values) & np.isfinite(opex_lp_upper_values)
_lower_violations = np.where(_fin & (opex_lp_lower_values > opex_milp_values * (1 + 1e-6)))[0]
_upper_violations = np.where(_fin & (opex_lp_upper_values < opex_milp_values * (1 - _MIP_GAP)))[0]
if len(_lower_violations):
    print(f"ERROR: LP lower bound exceeds MILP at {len(_lower_violations)} point(s):")
    for _i in _lower_violations:
        print(f"  r={price_ratios[_i]:.4f}  LP_lower={opex_lp_lower_values[_i]:,.2f}  MILP={opex_milp_values[_i]:,.2f}  diff={opex_lp_lower_values[_i]-opex_milp_values[_i]:+,.2f}")
if len(_upper_violations):
    print(f"ERROR: LP upper bound (min of both) is below MILP by more than MIP gap ({_MIP_GAP:.0%}) at {len(_upper_violations)} point(s):")
    for _i in _upper_violations:
        rel = (opex_milp_values[_i] - opex_lp_upper_values[_i]) / opex_milp_values[_i]
        print(f"  r={price_ratios[_i]:.4f}  LP_upper={opex_lp_upper_values[_i]:,.2f}  MILP={opex_milp_values[_i]:,.2f}  diff={opex_lp_upper_values[_i]-opex_milp_values[_i]:+,.2f}  ({rel:.2%})")


def _plot_lp_upper(axis, ms: float = MS) -> None:
    """Faint individual modes + bold best-of-two line."""
    axis.plot(price_ratios, opex_lp_upper_bo_values,
              color=C_LP_UB_BO,  linewidth=0.9, alpha=0.6, linestyle=(0, (5, 3)),
              label=r"$LP^U_B$")
    axis.plot(price_ratios, opex_lp_upper_chp_values,
              color=C_LP_UB_CHP, linewidth=0.9, alpha=0.6, linestyle=(0, (2, 2)),
              label=r"$LP^U_{CHP}$")
    axis.plot(price_ratios, opex_lp_upper_values,
              color=C_LP_UB_MIN, linewidth=1.5, linestyle="-", marker="^", markersize=ms,
              label=r"$LP^U$")


def _plot_main_series(axis, ms: float = MS) -> None:
    """The four headline curves (MILP, LP lower, LP upper min, LP approx) plus the two faint
    LP-upper heuristics, all sharing one marker-size scale so a main panel and its insets look
    like the same plot at different zoom."""
    axis.plot(price_ratios, opex_milp_values, color=C_MILP, linewidth=1.8, linestyle="-",
              marker="o", markersize=ms, label="MILP")
    axis.plot(price_ratios, opex_lp_lower_values, color=C_LP_LOWER, linewidth=1.5,
              linestyle=(0, (4, 2)), marker="s", markersize=ms, label=r"$LP^L$")
    _plot_lp_upper(axis, ms=ms)
    axis.plot(price_ratios, opex_lp_approx_mean_values, color=C_LP_APPROX, linewidth=1.5,
              linestyle=(0, (3, 1, 1, 1)), marker="*", markersize=ms + 1,
              label=r"$LP^{approx}$")


# The four headline curves an inset's y-limits are sized to -- not the two faint LP-upper
# heuristics, which are shown for context but are not what "the plot" is about.
_MAIN_SERIES = [opex_milp_values, opex_lp_lower_values, opex_lp_upper_values,
                opex_lp_approx_mean_values]


def _inset_ylim(xlim: tuple[float, float], log: bool, extra_top: float = 0.0) -> tuple[float, float]:
    """y-limits spanning all four headline curves within xlim, so an inset never crops one of
    them out -- the failure mode of sizing the range from only some of the series.

    extra_top: additional headroom as a fraction of the span, on top of the usual padding --
    for insets that also show the faint LP-upper heuristics (bo/chp), which sit above the
    min-of-both line the range is otherwise sized to and would get cut off without it.
    """
    mask = (price_ratios >= xlim[0]) & (price_ratios <= xlim[1])
    vals = np.concatenate([s[mask] for s in _MAIN_SERIES])
    vals = vals[np.isfinite(vals)]
    lo, hi = vals.min(), vals.max()
    if log:
        return lo * 0.96, hi * (1.04 + extra_top)
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad + extra_top * (hi - lo)


def _add_inset(ax, rect: list[float], xlim: tuple[float, float], title: str, log: bool,
                extra_top: float = 0.0):
    axins = ax.inset_axes(rect)
    _plot_main_series(axins, ms=MS + 0.5)
    if log:
        axins.set_xscale("log")
        axins.set_yscale("log")
    else:
        # The log insets' ticks are already sparse (log-spaced); the linear ones default to as
        # many major ticks as fit, which is cluttered at this inset size.
        axins.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axins.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axins.set_xlim(*xlim)
    axins.set_ylim(*_inset_ylim(xlim, log=log, extra_top=extra_top))
    axins.tick_params(axis="both", which="both", labelsize=5, pad=1)
    axins.set_title(title, fontsize=7, pad=8)
    ax.indicate_inset_zoom(axins, edgecolor="0.4")
    return axins


# Low-r inset: leftmost decade tenth of the log span, in both figures the same absolute window.
INS_X_LO = price_ratios[0]
INS_X_HI = price_ratios[0] * (price_ratios[-1] / price_ratios[0]) ** 0.10
INS_TITLE = f"low-$r$ zoom ($r≤{INS_X_HI:.2f}$)"

# Transition-region inset: r in [0.8, 1.1], around the CHP/boiler break-even at r=1.
INS2_X_LO, INS2_X_HI = 0.8, 1.1
INS2_TITLE = r"zoom $r\in[0.8,\,1.1]$"


def plot_linear() -> None:
    apply_style(width_cm=16, aspect="golden", grid=True, strict=True)
    fig, ax = plt.subplots(constrained_layout=True)

    _plot_main_series(ax)
    ax.set_xlabel(r"Price ratio $c_{\mathrm{gas}}\,/\,c_{\mathrm{el}}$ $[-]$")
    ax.set_ylabel(OPEX_YLABEL)
    ax.set_title("OPEX vs Price Ratio (1D angle training set)")
    ax.legend()
    ax.grid(alpha=0.25, linewidth=0.4)

    _add_inset(ax, [0.58, 0.07, 0.25, 0.3], (INS_X_LO, INS_X_HI), INS_TITLE, log=False,
               extra_top=0.25)
    _add_inset(ax, [0.25, 0.58, 0.25, 0.3], (INS2_X_LO, INS2_X_HI), INS2_TITLE, log=False)

    fig.savefig(OUT_DIR / "opex_vs_price_ratio_linear.png")
    fig.savefig(OUT_DIR / "opex_vs_price_ratio_linear.pdf")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'opex_vs_price_ratio_linear.png'} (+ .pdf)")


def plot_log() -> None:
    apply_style(width_cm=16, aspect="golden", grid=True, strict=True)
    fig, ax = plt.subplots(constrained_layout=True)

    _plot_main_series(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Price ratio $c_{\mathrm{gas}}\,/\,c_{\mathrm{el}}$ $[-]$")
    ax.set_ylabel(OPEX_YLABEL)

    gap_mask   = (np.isfinite(opex_milp_values) & np.isfinite(opex_lp_lower_values)
                  & np.isfinite(opex_lp_upper_values) & (opex_milp_values > 0))
    r_masked   = price_ratios[gap_mask]
    milp_m     = opex_milp_values[gap_mask]
    lower_gaps = (milp_m - opex_lp_lower_values[gap_mask]) / milp_m
    upper_gaps = (opex_lp_upper_values[gap_mask] - milp_m) / milp_m
    wi_lo, wi_up = np.argmax(lower_gaps), np.argmax(upper_gaps)
    # two lines: on one line this does not fit across the axes at the style's figure width
    gap_subtitle = (
        f"mean $\\Delta$: LP lower: {lower_gaps.mean():.2%}  |  LP upper: {upper_gaps.mean():.2%}\n"
        f"worst $\\Delta$: LP lower: {lower_gaps[wi_lo]:.2%} at $r={r_masked[wi_lo]:.3f}$"
        f"  |  LP upper: {upper_gaps[wi_up]:.2%} at $r={r_masked[wi_up]:.3f}$"
    )
    ax.set_title("OPEX vs Price Ratio — log scale", pad=24)
    ax.text(0.5, 1.0, gap_subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7, linespacing=1.4)
    ax.legend()
    ax.grid(alpha=0.25, linewidth=0.4)

    _add_inset(ax, [0.56, 0.05, 0.41, 0.35], (INS_X_LO, INS_X_HI), INS_TITLE, log=True)
    _add_inset(ax, [0.3, 0.60, 0.3, 0.33], (INS2_X_LO, INS2_X_HI), INS2_TITLE, log=True)

    fig.savefig(OUT_DIR / "opex_vs_price_ratio_log.png")
    fig.savefig(OUT_DIR / "opex_vs_price_ratio_log.pdf")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'opex_vs_price_ratio_log.png'} (+ .pdf)")


if __name__ == "__main__":
    plot_linear()
    plot_log()
