"""Sum of MILP commitment deltas over time steps, across the price ratios the 1D_angle
surrogate was trained on.

Unrelated to the dispatch plot scripts (plot_milp_dispatch.py etc.), which plot a single
solved dispatch at one price ratio. This instead solves the MILP at every ratio r = c_G / c_el
in the 1D_angle training set and sums its boiler and CHP commitment deltas (dB1+dB2,
dCHP1+dCHP2) over all time steps k, then plots the two sums against the price ratio -- showing
where the optimal commitment switches from CHP-mode to boiler-mode.

    python Marius/visualization/plot_dispatch_deltas.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.optimization.core import solve_milp
from src.visualization.style import apply_style

MIP_GAP = 1e-2
STRICT_DEMAND_SATISFACTION = True
C_EL = 1.0

_demand_df = pd.read_csv(ROOT / "energy_demands.csv")
Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()

# Exactly the 40 angle-sampled training ratios the 1D_angle surrogate was fitted on
# (create_sample("angle", 40)), read straight from that training set so this plot sits on the
# same ratios as the regression.
RATIO_SOURCE_CSV = ROOT / "Marius" / "results" / "evaluation_40_training_samples_1D_angle.csv"
DATA_CSV = ROOT / "Marius" / "results" / "delta_sums_vs_price_ratio_angle_40.csv"
OUT_STEM = ROOT / "Marius" / "visualization" / "delta_sums_vs_price_ratio"

# Manually rerun the sweep by setting this to True
REWRITE = False

C_DB = "#1B7837"    # dark green
C_DCHP = "#E08214"  # warm orange
MS = 2.5


def solve_delta_sums(price_ratios: np.ndarray) -> pd.DataFrame:
    """Solve the MILP at each ratio and sum its boiler/CHP commitment deltas over time."""
    sum_dB, sum_dCHP = [], []
    for r in price_ratios:
        c_g = C_EL * r
        print(f"c_G={c_g:.4f}  c_el={C_EL:.4f}  ratio={r:.4f}")
        _, dispatch = solve_milp(Q_D, P_D, c_g, C_EL, mip_gap=MIP_GAP,
                                 strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION)
        sum_dB.append((dispatch["dB1"] + dispatch["dB2"]).sum())
        sum_dCHP.append((dispatch["dCHP1"] + dispatch["dCHP2"]).sum())
    return pd.DataFrame({"price_ratio": price_ratios, "sum_dB": sum_dB, "sum_dCHP": sum_dCHP})


def delta_sums() -> pd.DataFrame:
    """The delta sums this plot is built on: reused from disk if the angle ratio sweep has
    already been solved, otherwise solved now and saved so the next run can reuse it."""
    if not RATIO_SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Ratio source {RATIO_SOURCE_CSV} not found. It is the 1D_angle training set "
            f"produced by Marius/surrogate_models/run_full_pipeline.py (N_TRAIN=40)."
        )
    price_ratios = pd.read_csv(RATIO_SOURCE_CSV)["ratio"].to_numpy()
    print(f"Sweeping the {len(price_ratios)} angle-sampled training ratios from "
          f"{RATIO_SOURCE_CSV} (r = {price_ratios.min():.4f} .. {price_ratios.max():.4f})")

    if DATA_CSV.exists() and not REWRITE:
        cached = pd.read_csv(DATA_CSV)
        if (len(cached) == len(price_ratios)
                and np.allclose(cached["price_ratio"].to_numpy(), price_ratios)):
            print(f"Reusing cached delta sums from {DATA_CSV}")
            return cached
        print(f"Cached {DATA_CSV} is for a different ratio sweep -- re-solving.")

    df = solve_delta_sums(price_ratios)
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    print(f"Saved delta sums to {DATA_CSV}")
    return df


def main():
    df = delta_sums()
    price_ratios = df["price_ratio"].to_numpy()

    apply_style(width_cm=16, aspect=2.6, grid=True, strict=True)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(price_ratios, df["sum_dB"], color=C_DB, linewidth=1.5, marker="^",
            markersize=MS, label=r"$\sum_{i,k}\,\delta_{\mathrm{B},i,k}$")
    ax.plot(price_ratios, df["sum_dCHP"], color=C_DCHP, linewidth=1.5, marker="v",
            markersize=MS, label=r"$\sum_{i,k}\,\delta_{\mathrm{CHP},i,k}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Price ratio $c_{\mathrm{gas}}\,/\,c_{\mathrm{el}}$ $[-]$")
    ax.set_ylabel(r"$\sum_{i,k}\,\delta_{i,k}\;[-]$")
    ax.legend()

    # Fainter than the style's default grid, and extended to both axes' minor ticks too --
    # here the grid is only a reading aid behind the two delta-sum curves.
    ax.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.3)

    ax.axvline(0.7, color="gray", linewidth=1.2, linestyle="--")
    ax.axvline(1.01, color="gray", linewidth=1.2, linestyle="--")
    y_lo, y_hi = ax.get_ylim()
    y_mid = 1.1 * (y_lo + y_hi) / 2
    x_chp_mid = np.sqrt(price_ratios[0] * 0.7)       # geometric centre of CHP region
    x_be_mid = np.sqrt(0.7 * 1.01)                   # geometric centre of break-even region
    x_boiler_mid = np.sqrt(1.01 * price_ratios[-1])  # geometric centre of boiler region
    ax.text(x_chp_mid, y_mid, "CHP-mode", ha="center", va="center",
            color="dimgray", fontweight="bold")
    ax.text(x_boiler_mid, y_mid, "Boiler-mode", ha="center", va="center",
            color="dimgray", fontweight="bold")
    ax.annotate(
        "Break-even",
        xy=(x_be_mid, y_mid),
        xytext=(np.sqrt(x_boiler_mid), y_hi * 0.78),
        ha="center", va="center", color="dimgray", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    )

    out_png = OUT_STEM.with_suffix(".png")
    out_pdf = OUT_STEM.with_suffix(".pdf")
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved figure to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
