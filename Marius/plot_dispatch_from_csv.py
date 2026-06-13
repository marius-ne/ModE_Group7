"""Plot dispatch results from a CSV export for LP or MILP runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from formulation_MILP import plot_dispatch_results as plot_dispatch_results_milp
from formulation_LP import plot_dispatch_results as plot_dispatch_results_lp


QOUT_NOM_B = 530.0
ETA_NOM_B = 0.8
QOUT_NOM_CHP = 470.0
ETA_NOM_CHP_TH = 0.481


def _unit_profile(dispatch: pd.DataFrame, mode: str) -> np.ndarray:
    """Return a 4 x N normalized unit profile in [0, 1]."""
    mode = mode.upper()
    if mode == "MILP":
        return np.vstack([
            np.clip(dispatch["dB1"].to_numpy(), 0.0, 1.0),
            np.clip(dispatch["dB2"].to_numpy(), 0.0, 1.0),
            np.clip(dispatch["dCHP1"].to_numpy(), 0.0, 1.0),
            np.clip(dispatch["dCHP2"].to_numpy(), 0.0, 1.0),
        ])

    qin_max_b = QOUT_NOM_B / ETA_NOM_B
    qin_max_chp = QOUT_NOM_CHP / ETA_NOM_CHP_TH
    return np.vstack([
        np.clip(dispatch["Qin_B1"].to_numpy() / qin_max_b, 0.0, 1.0),
        np.clip(dispatch["Qin_B2"].to_numpy() / qin_max_b, 0.0, 1.0),
        np.clip(dispatch["Qin_CHP1"].to_numpy() / qin_max_chp, 0.0, 1.0),
        np.clip(dispatch["Qin_CHP2"].to_numpy() / qin_max_chp, 0.0, 1.0),
    ])


def _validate_columns(dispatch: pd.DataFrame, mode: str) -> None:
    common_cols = {
        "k",
        "Q_D",
        "P_D",
        "E_TES",
        "Qin_TES",
        "Qout_TES",
        "Pgrid",
        "Qin_B1",
        "Qin_B2",
        "Qout_B1",
        "Qout_B2",
        "Qin_CHP1",
        "Qin_CHP2",
        "Qout_CHP1",
        "Qout_CHP2",
        "Pout_CHP1",
        "Pout_CHP2",
    }
    missing = sorted(common_cols.difference(dispatch.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if mode == "MILP":
        milp_cols = {"dB1", "dB2", "dCHP1", "dCHP2"}
        missing_milp = sorted(milp_cols.difference(dispatch.columns))
        if missing_milp:
            raise ValueError(f"MILP mode requires commitment columns: {missing_milp}")


def plot_dispatch_from_csv(
    csv_path: str,
    mode: str,
    output_path: str | None = None,
    gas_price: float | None = None,
    el_price: float | None = None,
) -> Path:
    """Create a dispatch overview plot from a dispatch CSV using the formulation plotters."""
    mode = mode.upper()
    if mode not in {"LP", "MILP"}:
        raise ValueError("mode must be 'LP' or 'MILP'")

    csv_file = Path(csv_path)
    dispatch = pd.read_csv(csv_file)
    _validate_columns(dispatch, mode)

    if output_path is None:
        output_file = Path("Marius/visualization") / f"dispatch_overview_from_csv_{mode}.png"
    else:
        output_file = Path(output_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if mode == "MILP":
        plot_dispatch_results_milp(
            dispatch,
            output_path=str(output_file),
            gas_price=gas_price,
            el_price=el_price,
        )
    else:
        plot_dispatch_results_lp(
            dispatch,
            output_path=str(output_file),
            gas_price=gas_price,
            el_price=el_price,
        )
    return output_file


def plot_dispatch_comparison_from_csv(
    csv_path_a: str,
    mode_a: str,
    csv_path_b: str,
    mode_b: str,
    output_path: str | None = None,
    label_a: str = "Solution A",
    label_b: str = "Solution B",
    gas_price: float | None = None,
    el_price: float | None = None,
) -> Path:
    """Create an overlay comparison plot for two dispatch CSV files."""
    mode_a = mode_a.upper()
    mode_b = mode_b.upper()
    if mode_a not in {"LP", "MILP"} or mode_b not in {"LP", "MILP"}:
        raise ValueError("mode_a/mode_b must be 'LP' or 'MILP'")

    csv_a = Path(csv_path_a)
    csv_b = Path(csv_path_b)
    dispatch_a = pd.read_csv(csv_a)
    dispatch_b = pd.read_csv(csv_b)

    _validate_columns(dispatch_a, mode_a)
    _validate_columns(dispatch_b, mode_b)

    k_a = dispatch_a["k"].to_numpy()
    k_b = dispatch_b["k"].to_numpy()
    if len(k_a) != len(k_b) or not np.allclose(k_a, k_b):
        raise ValueError("Both CSVs must have the same time index 'k' to compare overlays.")

    if output_path is None:
        output_file = Path("Marius/visualization") / f"dispatch_comparison_{mode_a}_vs_{mode_b}.png"
    else:
        output_file = Path(output_path)

    k = k_a
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True, constrained_layout=True)

    # 1) Unit commitment/utilization comparison (mean over all units)
    unit_a = _unit_profile(dispatch_a, mode_a).mean(axis=0)
    unit_b = _unit_profile(dispatch_b, mode_b).mean(axis=0)
    axes[0].plot(k, unit_a, color="#238443", linewidth=2.0, label=f"{label_a} ({mode_a})")
    axes[0].plot(k, unit_b, color="#1F78B4", linewidth=2.0, linestyle="--", label=f"{label_b} ({mode_b})")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Mean unit level [-]")
    axes[0].set_title("Unit Commitment / Utilization Comparison")
    axes[0].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[0].legend(loc="upper right")

    # 2) TES operation comparison
    axes[1].plot(k, dispatch_a["Qout_TES"], color="#2C7FB8", linewidth=1.8, label=f"TES discharge {label_a}")
    axes[1].plot(k, dispatch_b["Qout_TES"], color="#2C7FB8", linewidth=1.8, linestyle="--", label=f"TES discharge {label_b}")
    axes[1].plot(k, -dispatch_a["Qin_TES"], color="#EF3B2C", linewidth=1.8, label=f"TES charge {label_a}")
    axes[1].plot(k, -dispatch_b["Qin_TES"], color="#EF3B2C", linewidth=1.8, linestyle="--", label=f"TES charge {label_b}")
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel("TES power [kW]")
    axes[1].set_title("TES Operation")
    axes[1].grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

    ax1_twin = axes[1].twinx()
    ax1_twin.plot(k, dispatch_a["E_TES"], color="#6A3D9A", linewidth=1.8, label=f"TES energy {label_a}")
    ax1_twin.plot(k, dispatch_b["E_TES"], color="#6A3D9A", linewidth=1.8, linestyle="--", label=f"TES energy {label_b}")
    ax1_twin.set_ylabel("TES energy [kWh]")

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right", ncol=2)

    # 3) Electrical supply mix comparison
    p_chp_total_a = dispatch_a["Pout_CHP1"] + dispatch_a["Pout_CHP2"]
    p_chp_total_b = dispatch_b["Pout_CHP1"] + dispatch_b["Pout_CHP2"]
    axes[2].plot(k, dispatch_a["Pgrid"], color="#FDAE61", linewidth=1.8, label=f"Grid import {label_a}")
    axes[2].plot(k, dispatch_b["Pgrid"], color="#FDAE61", linewidth=1.8, linestyle="--", label=f"Grid import {label_b}")
    axes[2].plot(k, p_chp_total_a, color="#1B9E77", linewidth=1.8, label=f"CHP electric {label_a}")
    axes[2].plot(k, p_chp_total_b, color="#1B9E77", linewidth=1.8, linestyle="--", label=f"CHP electric {label_b}")
    axes[2].plot(k, dispatch_a["P_D"], color="#111111", linewidth=1.4, linestyle=":", label="Electric demand")
    axes[2].set_title("Electrical Supply Mix")
    axes[2].set_ylabel("Power [kW]")
    axes[2].set_xlabel("Time step k [-]")
    axes[2].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[2].legend(loc="upper right", ncol=2)

    # 4) Heat supply and gas purchase comparison
    q_boiler_total_a = dispatch_a["Qout_B1"] + dispatch_a["Qout_B2"]
    q_chp_total_a = dispatch_a["Qout_CHP1"] + dispatch_a["Qout_CHP2"]
    q_tes_net_a = dispatch_a["Qout_TES"] - dispatch_a["Qin_TES"]
    q_gas_total_a = dispatch_a["Qin_B1"] + dispatch_a["Qin_B2"] + dispatch_a["Qin_CHP1"] + dispatch_a["Qin_CHP2"]

    q_boiler_total_b = dispatch_b["Qout_B1"] + dispatch_b["Qout_B2"]
    q_chp_total_b = dispatch_b["Qout_CHP1"] + dispatch_b["Qout_CHP2"]
    q_tes_net_b = dispatch_b["Qout_TES"] - dispatch_b["Qin_TES"]
    q_gas_total_b = dispatch_b["Qin_B1"] + dispatch_b["Qin_B2"] + dispatch_b["Qin_CHP1"] + dispatch_b["Qin_CHP2"]

    axes[3].plot(k, q_boiler_total_a, color="#E31A1C", linewidth=1.8, label=f"Boiler heat {label_a}")
    axes[3].plot(k, q_boiler_total_b, color="#E31A1C", linewidth=1.8, linestyle="--", label=f"Boiler heat {label_b}")
    axes[3].plot(k, q_chp_total_a, color="#FF7F00", linewidth=1.8, label=f"CHP heat {label_a}")
    axes[3].plot(k, q_chp_total_b, color="#FF7F00", linewidth=1.8, linestyle="--", label=f"CHP heat {label_b}")
    axes[3].plot(k, q_tes_net_a, color="#2C7FB8", linewidth=1.8, label=f"TES net heat {label_a}")
    axes[3].plot(k, q_tes_net_b, color="#2C7FB8", linewidth=1.8, linestyle="--", label=f"TES net heat {label_b}")
    axes[3].plot(k, dispatch_a["Q_D"], color="#111111", linewidth=1.4, linestyle=":", label="Heat demand")
    axes[3].set_title("Heat Supply and Gas Purchase")
    axes[3].set_ylabel("Heat flow [kW]")
    axes[3].set_xlabel("Time step k [-]")
    axes[3].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    ax2_twin = axes[3].twinx()
    ax2_twin.plot(k, q_gas_total_a, color="#33A02C", linewidth=1.8, alpha=0.9, label=f"Gas purchased {label_a}")
    ax2_twin.plot(k, q_gas_total_b, color="#33A02C", linewidth=1.8, alpha=0.9, linestyle="--", label=f"Gas purchased {label_b}")
    ax2_twin.set_ylabel("Gas purchased [kW_fuel]")

    lines3, labels3 = axes[3].get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    axes[3].legend(lines3 + lines4, labels3 + labels4, loc="upper right", ncol=2)

    title = f"Operational Dispatch Comparison ({mode_a} vs {mode_b})"
    if gas_price is not None and el_price is not None:
        title += f" - gas={gas_price:.3f} €, el={el_price:.3f} €"
    fig.suptitle(title, fontsize=14)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot LP or MILP dispatch from CSV")
    parser.add_argument("--csv", required=True, help="Path to dispatch CSV file")
    parser.add_argument("--mode", required=True, choices=["LP", "MILP", "lp", "milp"], help="Result type")
    parser.add_argument("--csv-2", default=None, help="Optional second CSV for overlay comparison")
    parser.add_argument("--mode-2", default=None, choices=["LP", "MILP", "lp", "milp"], help="Result type for second CSV")
    parser.add_argument("--label-1", default="Solution A", help="Legend label for first solution")
    parser.add_argument("--label-2", default="Solution B", help="Legend label for second solution")
    parser.add_argument("--output", default=None, help="Optional output PNG path")
    parser.add_argument("--gas-price", type=float, default=None, help="Optional gas price for title")
    parser.add_argument("--el-price", type=float, default=None, help="Optional electricity price for title")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.csv_2 is not None:
        if args.mode_2 is None:
            raise ValueError("--mode-2 is required when --csv-2 is provided")
        output = plot_dispatch_comparison_from_csv(
            csv_path_a=args.csv,
            mode_a=args.mode,
            csv_path_b=args.csv_2,
            mode_b=args.mode_2,
            output_path=args.output,
            label_a=args.label_1,
            label_b=args.label_2,
            gas_price=args.gas_price,
            el_price=args.el_price,
        )
    else:
        output = plot_dispatch_from_csv(
            csv_path=args.csv,
            mode=args.mode,
            output_path=args.output,
            gas_price=args.gas_price,
            el_price=args.el_price,
        )
    print(f"Saved plot to: {output}")
