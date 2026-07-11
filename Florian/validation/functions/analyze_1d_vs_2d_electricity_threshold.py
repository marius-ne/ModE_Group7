from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualize_ratio_vs_absolute_regression_errors import (
    predict_absolute_opex,
    prepare_inputs,
    load_test_data,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT / "one_d_vs_two_d_error_by_electricity_price.csv"
OUTPUT_PNG = ROOT / "one_d_vs_two_d_electricity_threshold.png"
OUTPUT_DECISION_PNG = ROOT / "one_d_vs_two_d_price_region_decision.png"
OUTPUT_TXT = ROOT / "one_d_vs_two_d_electricity_threshold_summary.txt"

COLOR_2D = "tab:blue"
COLOR_1D = "tab:red"


def calculate_errors() -> pd.DataFrame:
    ratio_1d_df, absolute_2d_df = load_test_data()
    prepared = prepare_inputs(ratio_1d_df, absolute_2d_df)
    result = predict_absolute_opex(prepared)

    analysis = pd.DataFrame(
        {
            "electricity_price_MWh": result["c_el_raw"],
            "gas_price_MWh": result["c_G_raw"],
            "opex_true": result["opex_true"],
            "error_2d_absolute_regression": result["error_3d"],
            "error_1d_ratio_regression": result["error_2d"],
        }
    )
    analysis["advantage_2d_minus_1d"] = (
        analysis["error_1d_ratio_regression"] - analysis["error_2d_absolute_regression"]
    )
    analysis["relative_error_2d_absolute_regression"] = (
        analysis["error_2d_absolute_regression"] / analysis["opex_true"]
    )
    analysis["relative_error_1d_ratio_regression"] = (
        analysis["error_1d_ratio_regression"] / analysis["opex_true"]
    )
    analysis["relative_advantage_2d_minus_1d"] = (
        analysis["relative_error_1d_ratio_regression"] - analysis["relative_error_2d_absolute_regression"]
    )
    analysis["specific_opex_true"] = result["opex_true"] / result["c_el"]
    analysis["specific_opex_pred_2d"] = result["opex_pred_3d"] / result["c_el"]
    analysis["specific_opex_pred_1d"] = result["opex_pred_ratio_specific"]
    analysis["specific_error_2d_absolute_regression"] = (
        analysis["specific_opex_true"] - analysis["specific_opex_pred_2d"]
    ).abs()
    analysis["specific_error_1d_ratio_regression"] = (
        analysis["specific_opex_true"] - analysis["specific_opex_pred_1d"]
    ).abs()
    analysis["specific_advantage_1d_minus_2d"] = (
        analysis["specific_error_2d_absolute_regression"] - analysis["specific_error_1d_ratio_regression"]
    )
    analysis["better_model"] = np.where(
        analysis["advantage_2d_minus_1d"] > 0,
        "2D Absolute Regression",
        "1D Ratio Regression",
    )
    return analysis.sort_values("electricity_price_MWh").reset_index(drop=True)


def error_curve_crossing_threshold(df: pd.DataFrame) -> float:
    prices = df["electricity_price_MWh"].to_numpy()
    advantage = df["advantage_2d_minus_1d"].to_numpy()

    for idx in range(len(prices) - 1):
        y0 = advantage[idx]
        y1 = advantage[idx + 1]
        if np.isclose(y0, 0.0):
            return float(prices[idx])
        if y0 * y1 < 0:
            x0 = prices[idx]
            x1 = prices[idx + 1]
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))

    closest_idx = int(np.argmin(np.abs(advantage)))
    return float(prices[closest_idx])


def split_quality(df: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    prices = df["electricity_price_MWh"].to_numpy()
    two_d_better = (df["better_model"] == "2D Absolute Regression").to_numpy()
    predicted_two_d_better = prices >= threshold
    correct = int((predicted_two_d_better == two_d_better).sum())
    accuracy = correct / len(df)
    mean_advantage_high = df.loc[df["electricity_price_MWh"] >= threshold, "advantage_2d_minus_1d"].mean()
    mean_advantage_low = df.loc[df["electricity_price_MWh"] < threshold, "advantage_2d_minus_1d"].mean()
    return {
        "threshold": float(threshold),
        "correct": correct,
        "accuracy": float(accuracy),
        "mean_advantage_high": float(mean_advantage_high) if not np.isnan(mean_advantage_high) else np.nan,
        "mean_advantage_low": float(mean_advantage_low) if not np.isnan(mean_advantage_low) else np.nan,
    }


def dominance_threshold(df: pd.DataFrame) -> float | None:
    prices = df["electricity_price_MWh"].to_numpy()
    two_d_better = (df["better_model"] == "2D Absolute Regression").to_numpy()

    for idx, price in enumerate(prices):
        if two_d_better[idx:].all():
            return float(price)
    return None


def summarize_by_price_region(df: pd.DataFrame) -> pd.DataFrame:
    region_df = df.copy()
    region_df["price_region"] = pd.qcut(
        region_df["electricity_price_MWh"],
        q=3,
        labels=["low C_el", "medium C_el", "high C_el"],
    )
    summary = (
        region_df.groupby("price_region", observed=True)
        .agg(
            electricity_min_MWh=("electricity_price_MWh", "min"),
            electricity_max_MWh=("electricity_price_MWh", "max"),
            mean_error_2d=("error_2d_absolute_regression", "mean"),
            mean_error_1d=("error_1d_ratio_regression", "mean"),
            mean_advantage_2d=("advantage_2d_minus_1d", "mean"),
        )
        .reset_index()
    )
    summary["better_on_average"] = np.where(
        summary["mean_advantage_2d"] > 0,
        "2D Absolute Regression",
        "1D Ratio Regression",
    )
    return summary


def plot_threshold_analysis(
    df: pd.DataFrame,
    split: dict[str, float | int],
    dominance: float | None,
) -> Path:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 8), sharex=True)

    x = df["electricity_price_MWh"]
    advantage = df["advantage_2d_minus_1d"]
    point_colors = np.where(df["better_model"] == "2D Absolute Regression", COLOR_2D, COLOR_1D)

    axes[0].scatter(x, advantage, c=point_colors, edgecolors="black", s=90, zorder=3)
    axes[0].axhline(0, color="black", linewidth=1.0, linestyle="--")
    axes[0].axvline(split["threshold"], color="0.2", linewidth=1.5, linestyle="-", label="best split threshold")

    axes[0].fill_between(
        [x.min(), split["threshold"]],
        advantage.min(),
        advantage.max(),
        color=COLOR_1D,
        alpha=0.08,
        label="rule region: 1D preferred",
    )
    axes[0].fill_between(
        [split["threshold"], x.max()],
        advantage.min(),
        advantage.max(),
        color=COLOR_2D,
        alpha=0.08,
        label="rule region: 2D preferred",
    )
    axes[0].set_ylabel("Error advantage of 2D [EUR]\npositive = 2D better")
    axes[0].set_title(
        "LR MILP: where does the 2D absolute regression become better than the 1D ratio regression?"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(
        x,
        df["error_2d_absolute_regression"],
        color=COLOR_2D,
        marker="o",
        linewidth=1.8,
        label="2D Absolute Regression",
    )
    axes[1].plot(
        x,
        df["error_1d_ratio_regression"],
        color=COLOR_1D,
        marker="o",
        linewidth=1.8,
        label="1D Ratio Regression",
    )
    axes[1].axvline(split["threshold"], color="0.2", linewidth=1.5, linestyle="-")
    axes[1].set_xlabel("Electricity price C_el [EUR/MWh]")
    axes[1].set_ylabel("Absolute OPEX error [EUR]")
    axes[1].set_title("LR MILP absolute OPEX prediction error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("1D vs 2D regression threshold analysis for LR MILP", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PNG


def plot_price_region_decision(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.5))

    x = df["electricity_price_MWh"].to_numpy()
    y = df["gas_price_MWh"].to_numpy()
    z = df["advantage_2d_minus_1d"].to_numpy()
    point_colors = np.where(df["better_model"] == "2D Absolute Regression", COLOR_2D, COLOR_1D)

    triangulation = mtri.Triangulation(x, y)
    max_abs = max(abs(z.min()), abs(z.max()))
    contour = ax.tricontourf(
        triangulation,
        z,
        levels=np.linspace(-max_abs, max_abs, 13),
        cmap="RdBu",
        alpha=0.35,
        extend="both",
    )
    ax.tricontour(triangulation, z, levels=[0], colors="black", linewidths=1.7, linestyles="--")

    ax.scatter(x, y, c=point_colors, edgecolors="black", s=95, zorder=3)
    for _, row in df.iterrows():
        label = "2D" if row["better_model"] == "2D Absolute Regression" else "1D"
        ax.annotate(
            label,
            (row["electricity_price_MWh"], row["gas_price_MWh"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Error advantage of 2D [EUR]\npositive = 2D better")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_2D, markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_1D, markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.7),
    ]
    ax.legend(handles, ["2D better", "1D better", "equal-error boundary"], loc="best")
    ax.set_xlabel("Electricity price C_el [EUR/MWh]")
    ax.set_ylabel("Gas price C_gas [EUR/MWh]")
    ax.set_title("Price regions where 1D ratio or 2D absolute regression has lower OPEX error")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DECISION_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_DECISION_PNG


def write_summary(
    split: dict[str, float | int],
    dominance: float | None,
    region_summary: pd.DataFrame,
) -> None:
    lines = [
        "1D vs 2D electricity-price threshold analysis",
        "",
        f"Best split threshold: C_el >= {split['threshold']:.2f} EUR/MWh -> use 2D",
        f"Correct decisions with this simple rule: {split['correct']} / 10 ({split['accuracy']:.0%})",
    ]
    if dominance is None:
        lines.append("No observed C_el threshold where 2D is better for every higher-price sample.")
    else:
        lines.append(f"Strict observed dominance threshold: C_el >= {dominance:.2f} EUR/MWh")
    lines.append("Best split threshold is the piecewise-linear crossing of the absolute-error curves.")

    lines.extend(["", "Price-region averages:", region_summary.to_string(index=False)])
    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = calculate_errors()
    crossing_threshold = error_curve_crossing_threshold(df)
    split = split_quality(df, crossing_threshold)
    dominance = dominance_threshold(df)
    region_summary = summarize_by_price_region(df)

    df.to_csv(OUTPUT_CSV, index=False)
    plot_path = plot_threshold_analysis(df, split, dominance)
    decision_plot_path = plot_price_region_decision(df)
    write_summary(split, dominance, region_summary)

    print(df)
    print()
    print(f"Best split threshold: C_el >= {split['threshold']:.2f} EUR/MWh -> use 2D")
    print(f"Correct decisions: {split['correct']} / {len(df)} ({split['accuracy']:.0%})")
    if dominance is not None:
        print(f"Strict observed dominance threshold: C_el >= {dominance:.2f} EUR/MWh")
    else:
        print("No strict observed dominance threshold found.")
    print(f"Saved point-wise CSV to: {OUTPUT_CSV}")
    print(f"Saved summary to: {OUTPUT_TXT}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved price-region decision plot to: {decision_plot_path}")


if __name__ == "__main__":
    main()
