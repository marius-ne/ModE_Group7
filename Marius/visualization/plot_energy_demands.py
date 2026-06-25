"""Time series and sorted load duration curves for heat and electricity demand."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


_BLUE  = "#1565C0"
_RED   = "#C62828"


def plot_energy_demands(
    csv_path: str | None = None,
    output_dir: str | None = None,
    fontsize: int = 11,
) -> Path:
    csv  = Path(csv_path)  if csv_path  else Path("energy_demands.csv")
    base = Path(output_dir) if output_dir else Path("Marius/visualization")
    base.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    df.columns = ["time_h", "power_kw", "heat_kw"]

    y_min = min(df["power_kw"].min(), df["heat_kw"].min()) * 0.9
    y_max = 950

    fig, (ax_ts, ax_sd) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: time series ─────────────────────────────────────────────────────
    ax_ts.plot(df["time_h"], df["power_kw"], color=_BLUE, linewidth=0.6,
               label="Electricity demand")
    ax_ts.plot(df["time_h"], df["heat_kw"],  color=_RED,  linewidth=0.6,
               label="Heat demand")
    ax_ts.set_xlabel("Time (h)", fontsize=fontsize)
    ax_ts.set_ylabel("Demand (kW)", fontsize=fontsize)
    ax_ts.set_title("Demand time series", fontsize=fontsize, fontweight="bold")
    ax_ts.set_ylim(y_min, y_max)
    ax_ts.legend(fontsize=fontsize - 1, loc="upper right")
    ax_ts.tick_params(labelsize=fontsize - 1)

    # ── Right: sorted load duration curve ─────────────────────────────────────
    hours = range(1, len(df) + 1)
    ax_sd.plot(hours, sorted(df["power_kw"], reverse=True), color=_BLUE,
               linewidth=0.8, label="Electricity demand")
    ax_sd.plot(hours, sorted(df["heat_kw"],  reverse=True), color=_RED,
               linewidth=0.8, label="Heat demand")
    ax_sd.set_xlabel("Hours (sorted)", fontsize=fontsize)
    ax_sd.set_ylabel("Demand (kW)", fontsize=fontsize)
    ax_sd.set_title("Load duration curves", fontsize=fontsize, fontweight="bold")
    ax_sd.set_ylim(y_min, y_max)
    ax_sd.legend(fontsize=fontsize - 1)
    ax_sd.tick_params(labelsize=fontsize - 1)

    fig.tight_layout()

    out_path = base / "energy_demands.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = plot_energy_demands()
    print(f"Saved → {out}")
