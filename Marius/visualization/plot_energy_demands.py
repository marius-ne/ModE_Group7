"""Time series and sorted load duration curves for heat and electricity demand."""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.visualization.style import apply_style

_BLUE  = "#1565C0"
_RED   = "#C62828"


def plot_energy_demands(
    csv_path: str | None = None,
    output_dir: str | None = None,
) -> Path:
    csv  = Path(csv_path)  if csv_path  else Path("energy_demands.csv")
    base = Path(output_dir) if output_dir else Path("Marius/visualization")
    base.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    df.columns = ["time_h", "power_kw", "heat_kw"]

    y_min = min(df["power_kw"].min(), df["heat_kw"].min()) * 0.9
    y_max = 950

    apply_style(width_cm=22, aspect=2.6, grid=True, strict=True)
    fig, (ax_ts, ax_sd) = plt.subplots(1, 2, constrained_layout=True)

    # ── Left: time series ─────────────────────────────────────────────────────
    ax_ts.plot(df["time_h"], df["power_kw"], color=_BLUE, linewidth=0.6,
               label="Electricity demand")
    ax_ts.plot(df["time_h"], df["heat_kw"],  color=_RED,  linewidth=0.6,
               label="Heat demand")
    ax_ts.set_xlabel("Time (h)")
    ax_ts.set_ylabel("Demand (kW)")
    ax_ts.set_title("Demand time series")
    ax_ts.set_ylim(y_min, y_max)
    ax_ts.legend(loc="upper right")

    # ── Right: sorted load duration curve ─────────────────────────────────────
    hours = range(1, len(df) + 1)
    ax_sd.plot(hours, sorted(df["power_kw"], reverse=True), color=_BLUE,
               linewidth=0.8, label="Electricity demand")
    ax_sd.plot(hours, sorted(df["heat_kw"],  reverse=True), color=_RED,
               linewidth=0.8, label="Heat demand")
    ax_sd.set_xlabel("Hours (sorted)")
    ax_sd.set_ylabel("Demand (kW)")
    ax_sd.set_title("Load duration curves")
    ax_sd.set_ylim(y_min, y_max)
    ax_sd.legend()

    out_path = base / "energy_demands.png"
    fig.savefig(out_path)  # dpi/bbox come from apply_style's rcParams
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = plot_energy_demands()
    print(f"Saved → {out}")
