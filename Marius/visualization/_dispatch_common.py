"""Shared pieces of the dispatch plot scripts, plus the on-disk format the run_*_dispatch.py
scripts write.

Split of responsibilities:

  Marius/evaluation/run_*_dispatch.py       solve_and_save() one price point; also usable
                                             standalone to solve+cache without plotting.
  Marius/visualization/plot_*_dispatch.py   call solve_and_save() themselves, then draw
                                             commitment + TES (dispatch_figure) and the energy
                                             balance (energy_balance_figure) as separate files.

plot_dispatch_deltas.py is unrelated to this module: it sweeps the MILP over a ratio range
and does not plot a single saved dispatch, so it does not import from here.

Nothing here imports from Marius/OUTDATED.

The commitment + TES figure (dispatch_figure) is composed from Erdem's own panel functions
in Erdem/src/visualization/core.py, so it comes out in the same format as Erdem's
results/Plots/dispatch.pdf. Erdem has no equivalent of the energy balance panels
(plot_electrical_mix, plot_heat_and_gas) -- they are written in the same style rather than
adapted from it. All panel functions here follow the same contract: take an `ax`, draw on it,
return it, and leave figure size/fonts/saving to apply_style and the caller.
"""

import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.visualization.core import plot_tes_operation, plot_unit_on_off_status
from src.visualization.style import apply_style, get_figsize

# Where run_*_dispatch.py writes its solved dispatch, and where the plot scripts read it from.
DISPATCH_DIR = ROOT / "Marius" / "results" / "dispatch"
FIGURE_DIR = ROOT / "Marius" / "visualization"

# The commitment variables of a dispatch, in the Marius column naming that
# Erdem/src/optimization/core.py's solvers return. Only the MILP (binary δ), the LP lower
# bound (δ relaxed to [0,1]) and the LP upper bound (δ fixed to a heuristic schedule, so also
# binary -- just chosen by a rule instead of optimized) have them; build_lp_approximated drops
# the commitment variables from the model entirely.
UNIT_COLS = ["dB1", "dB2", "dCHP1", "dCHP2"]
UNIT_NAMES = ["Boiler 1", "Boiler 2", "CHP 1", "CHP 2"]

FIG_WIDTH_CM = 16


# ---------------------------------------------------------------------------
# On-disk dispatch format
# ---------------------------------------------------------------------------
def ratio_from_argv(default: float) -> float:
    """The price ratio c_G/c_el to run or plot: the first command-line argument, else default.

    Both the run script and the plot scripts take it this way, so a ratio is named once per
    invocation and the two cannot drift apart -- the ratio is part of the filename, see
    dispatch_paths."""
    return float(sys.argv[1]) if len(sys.argv) > 1 else default


def dispatch_paths(formulation: str, ratio: float) -> tuple[Path, Path]:
    """(csv, json) the dispatch of `formulation` at this price ratio is stored under."""
    stem = f"dispatch_{formulation}_ratio{ratio:.3f}"
    return DISPATCH_DIR / f"{stem}.csv", DISPATCH_DIR / f"{stem}.json"


def save_dispatch(dispatch: pd.DataFrame, meta: dict, formulation: str, ratio: float) -> Path:
    """Write a solved dispatch and its metadata (prices, ratio, OPEX, solver settings).

    The metadata is what the plot scripts need for their titles and cannot recover from the
    dispatch table itself -- which is the whole reason it is saved alongside, rather than a
    solving script handing the numbers to a plotting function.
    """
    csv_path, json_path = dispatch_paths(formulation, ratio)
    DISPATCH_DIR.mkdir(parents=True, exist_ok=True)
    dispatch.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved dispatch to {csv_path}\nSaved metadata to {json_path}")
    return csv_path


def title_of(name: str, meta: dict) -> str:
    """Two-line figure title: which formulation, at which prices, for which OPEX."""
    return (f"{name}  —  $c_G$ = {meta['c_g']:.3f} €/kWh,  $c_{{el}}$ = {meta['c_el']:.3f} €/kWh"
            f"  |  $c_G/c_{{el}}$ = {meta['ratio']:.3f}\n"
            f"OPEX = {meta['opex']:,.2f} €")


def save_figure(fig, stem: str) -> Path:
    """Save to Marius/visualization/<stem>.pdf (dpi/bbox come from apply_style's rcParams)."""
    out = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return out


# ---------------------------------------------------------------------------
# The commitment + TES figure, adapted from Erdem's plot_dispatch_stacked
# ---------------------------------------------------------------------------
def to_erdem_columns(dispatch: pd.DataFrame) -> pd.DataFrame:
    """Rename the TES columns of a Marius-format dispatch to the names Erdem's
    plot_tes_operation reads. It falls back to zeros for columns it cannot find, so without
    this the TES panel would silently come out empty instead of failing."""
    return dispatch.rename(columns={"Qin_TES": "Q_in_TES", "Qout_TES": "Q_out_TES"})


def plot_unit_commitment_relaxed(dispatch: pd.DataFrame, ax=None) -> plt.Axes:
    """Commitment δ of each unit as a shaded timeline, for the LP lower bound -- the only
    formulation whose δ is genuinely continuous (relaxed to [0, 1]). Erdem's
    plot_unit_on_off_status thresholds at 0.5, which would throw the intermediate values away,
    so this gets a continuous colormap (and a colorbar) instead -- same layout otherwise."""
    if ax is None:
        _, ax = plt.subplots()

    k = dispatch["k"].to_numpy()
    delta = np.vstack([np.clip(dispatch[col].to_numpy(), 0.0, 1.0) for col in UNIT_COLS])

    cmap = mcolors.LinearSegmentedColormap.from_list("uc", ["#F3E7D3", "#238443"])
    im = ax.imshow(
        delta, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1,
        extent=[k[0] - 0.5, k[-1] + 0.5, -0.5, len(UNIT_COLS) - 0.5], origin="lower",
    )
    ax.set_yticks(np.arange(len(UNIT_COLS)))
    ax.set_yticklabels(UNIT_NAMES)
    ax.set_ylabel("Unit")
    ax.yaxis.set_minor_locator(plt.NullLocator())
    for i in range(1, len(UNIT_COLS)):
        ax.axhline(y=i - 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.figure.colorbar(im, ax=ax, label="$\\delta$ [0–1]", fraction=0.015, pad=0.01)
    return ax


# The figsize of Erdem's own MILP dispatch figure (Erdem/playground_analysis.ipynb:
# plot_dispatch_stacked(solution_df, figsize=(current_width, 4.5)), current_width being
# matplotlib's default figure width). Matched here exactly rather than derived from
# get_figsize/width_cm, so the two on/off + TES panels come out the same size as Erdem's. For
# the LP approximation (TES panel alone, no commitment row) the height is scaled down to the
# TES row's own share of that 4.5in, i.e. its height_ratios weight (2) out of the pair's (1+2).
MILP_DISPATCH_FIGSIZE = (6.4, 4.5)


def dispatch_figure(dispatch: pd.DataFrame, unit_panel: str | None) -> plt.Figure:
    """Erdem's stacked dispatch figure for a Marius dispatch: unit commitment on top, TES
    operation below, sharing one time axis (cf. Erdem's plot_dispatch_stacked). Untitled, like
    Erdem's own -- plot_dispatch_stacked never calls fig.suptitle either.

    unit_panel: "binary" for the MILP and the LP upper bound -- Erdem's own on/off panel,
    since both have true 0/1 δ (optimized for the MILP, fixed by a heuristic for LP upper);
    "relaxed" for the LP lower bound, the only one whose δ is genuinely continuous; or None
    for the LP approximation, which has no commitment variables at all and is therefore left
    with the TES panel alone.
    """
    n_rows = 1 if unit_panel is None else 2
    height_ratios = [1, 2][-n_rows:]
    width, full_height = MILP_DISPATCH_FIGSIZE
    figsize = (width, full_height * sum(height_ratios) / 3)
    apply_style(width_cm=FIG_WIDTH_CM, aspect="golden", nrows=n_rows, strict=True)
    fig, axes = plt.subplots(
        n_rows, 1, sharex=True, squeeze=False, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = list(axes[:, 0])

    if unit_panel == "binary":
        plot_unit_on_off_status(dispatch, ax=axes.pop(0), unit_cols=UNIT_COLS,
                                unit_names=UNIT_NAMES)
    elif unit_panel == "relaxed":
        plot_unit_commitment_relaxed(dispatch, ax=axes.pop(0))

    tes_ax = axes.pop(0)
    plot_tes_operation(to_erdem_columns(dispatch), ax=tes_ax)
    tes_ax.set_xlabel("Time step [h]")  # override Erdem's own "Time step [-]"

    fig.tight_layout()
    # Both panels carry their legend outside the axes (Erdem's do: on/off below, TES above),
    # so the rows need room between them that tight_layout does not leave on its own.
    fig.subplots_adjust(hspace=0.45)
    return fig


# ---------------------------------------------------------------------------
# The energy balance figure: where the electricity and heat come from
# ---------------------------------------------------------------------------
def plot_electrical_mix(dispatch: pd.DataFrame, ax=None) -> plt.Axes:
    """Where the electricity comes from: grid import (area) and the CHPs' output (line),
    against the electric demand they have to cover between them. Styled after the
    "Electrical Supply Mix" panel of Erdem's own plot_dispatch_results_compact."""
    if ax is None:
        _, ax = plt.subplots()

    k = dispatch["k"].to_numpy()
    p_chp = dispatch["Pout_CHP1"] + dispatch["Pout_CHP2"]

    ax.fill_between(k, 0, dispatch["Pgrid"], step="mid", alpha=0.45, color="#FDAE61",
                    label="Grid import")
    ax.plot(k, p_chp, color="#1B9E77", linewidth=2.0, label="CHP electrical output")
    ax.plot(k, dispatch["P_D"], color="#111111", linewidth=1.7, linestyle="--",
            label="Electrical demand")

    ax.set_title("Electrical Supply Mix")
    ax.set_ylabel("Power [kW]")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    return ax


def plot_heat_and_gas(dispatch: pd.DataFrame, ax=None) -> plt.Axes:
    """Where the heat comes from: boilers, CHPs and net TES discharge against the heat demand,
    with the gas bought to make it as bars behind them. Styled after the "Heat Supply and Gas
    Purchase" panel of Erdem's own plot_dispatch_results_compact."""
    if ax is None:
        _, ax = plt.subplots()

    k = dispatch["k"].to_numpy()
    q_boiler = dispatch["Qout_B1"] + dispatch["Qout_B2"]
    q_chp = dispatch["Qout_CHP1"] + dispatch["Qout_CHP2"]
    q_tes = dispatch["Qout_TES"] - dispatch["Qin_TES"]
    q_gas = dispatch["Qin_B1"] + dispatch["Qin_B2"] + dispatch["Qin_CHP1"] + dispatch["Qin_CHP2"]

    ax.bar(k, q_gas, width=0.85, alpha=0.22, color="#33A02C", label="Gas purchased")
    ax.plot(k, q_boiler, color="#E31A1C", linewidth=1.8, label="Boiler heat")
    ax.plot(k, q_chp, color="#FF7F00", linewidth=1.8, label="CHP heat")
    ax.plot(k, q_tes, color="#2C7FB8", linewidth=1.8, label="TES net heat")
    ax.plot(k, dispatch["Q_D"], color="#111111", linewidth=1.7, linestyle="--",
            label="Heat demand")

    ax.set_title("Heat Supply and Gas Purchase")
    ax.set_xlabel("Time step [h]")
    ax.set_ylabel("Heat flow / gas input [kW]")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    return ax


def energy_balance_figure(dispatch: pd.DataFrame, title: str) -> plt.Figure:
    """The electrical and heat/gas balances stacked on one shared time axis. Every formulation
    has an energy balance -- unlike the commitment δ, which the LP approximation does not have
    -- so this figure applies to all four."""
    apply_style(width_cm=FIG_WIDTH_CM, aspect="golden", nrows=2, strict=True)
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=get_figsize(FIG_WIDTH_CM, "golden", nrows=2))

    plot_electrical_mix(dispatch, ax=axes[0])
    plot_heat_and_gas(dispatch, ax=axes[1])

    fig.suptitle(title)
    fig.tight_layout()
    # Both panels carry their legend outside the axes to the right, like Erdem's own
    # plot_dispatch_results_compact does, so the plot area needs room on the right that
    # tight_layout does not leave on its own.
    fig.subplots_adjust(right=0.78, hspace=0.25)
    return fig
