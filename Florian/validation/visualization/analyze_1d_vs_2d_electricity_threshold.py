from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Florian.validation.visualization.visualize_ratio_vs_absolute_regression_errors import (
    predict_absolute_opex,
    prepare_inputs,
    load_test_data,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT / "one_d_vs_two_d_error_by_electricity_price.csv"
OUTPUT_PNG = ROOT / "one_d_vs_two_d_electricity_threshold.png"
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
    analysis["better_model"] = np.where(
        analysis["advantage_2d_minus_1d"] > 0,
        "2D Absolute Regression",
        "1D Ratio Regression",
    )
    return analysis.sort_values("electricity_price_MWh").reset_index(drop=True)


def best_split_threshold(df: pd.DataFrame) -> dict[str, float | int]:
    prices = df["electricity_price_MWh"].to_numpy()
    two_d_better = (df["better_model"] == "2D Absolute Regression").to_numpy()

    candidates = [(prices[i] + prices[i + 1]) / 2 for i in range(len(prices) - 1)]
    candidates = [prices[0] - 1e-9, *candidates, prices[-1] + 1e-9]

    best = None
    for threshold in candidates:
        predicted_two_d_better = prices >= threshold
        correct = int((predicted_two_d_better == two_d_better).sum())
        accuracy = correct / len(df)
        mean_advantage_high = df.loc[df["electricity_price_MWh"] >= threshold, "advantage_2d_minus_1d"].mean()
        mean_advantage_low = df.loc[df["electricity_price_MWh"] < threshold, "advantage_2d_minus_1d"].mean()

        candidate = {
            "threshold": float(threshold),
            "correct": correct,
            "accuracy": float(accuracy),
            "mean_advantage_high": float(mean_advantage_high) if not np.isnan(mean_advantage_high) else np.nan,
            "mean_advantage_low": float(mean_advantage_low) if not np.isnan(mean_advantage_low) else np.nan,
        }
        if best is None or (candidate["correct"], candidate["threshold"]) > (best["correct"], best["threshold"]):
            best = candidate

    return best


def dominance_threshold(df: pd.DataFrame) -> float | None:
    prices = df["electricity_price_MWh"].to_numpy()
    two_d_better = (df["better_model"] == "2D Absolute Regression").to_numpy()

    for idx, price in enumerate(prices):
        if two_d_better[idx:].all():
            return float(price)
    return None


def linear_crossing_threshold(df: pd.DataFrame) -> float | None:
    x = df["electricity_price_MWh"].to_numpy()
    y = df["advantage_2d_minus_1d"].to_numpy()
    slope, intercept = np.polyfit(x, y, deg=1)
    if np.isclose(slope, 0.0):
        return None

    crossing = -intercept / slope
    if x.min() <= crossing <= x.max():
        return float(crossing)
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
    crossing: float | None,
) -> Path:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 8), sharex=True)

    x = df["electricity_price_MWh"]
    advantage = df["advantage_2d_minus_1d"]
    point_colors = np.where(df["better_model"] == "2D Absolute Regression", COLOR_2D, COLOR_1D)

    axes[0].scatter(x, advantage, c=point_colors, edgecolors="black", s=90, zorder=3)
    axes[0].axhline(0, color="black", linewidth=1.0, linestyle="--")
    axes[0].axvline(split["threshold"], color="0.2", linewidth=1.5, linestyle="-", label="best split threshold")
    if dominance is not None:
        axes[0].axvline(dominance, color=COLOR_2D, linewidth=1.5, linestyle=":", label="2D dominates above")
    if crossing is not None:
        axes[0].axvline(crossing, color="tab:green", linewidth=1.5, linestyle="-.", label="linear crossing")

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
    axes[0].set_title("Where does the 2D absolute regression become better than the 1D ratio regression?")
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
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PNG


def write_summary(
    split: dict[str, float | int],
    dominance: float | None,
    crossing: float | None,
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
    if crossing is None:
        lines.append("Linear trend crossing: outside observed range or undefined.")
    else:
        lines.append(f"Linear trend crossing: C_el ~= {crossing:.2f} EUR/MWh")

    lines.extend(["", "Price-region averages:", region_summary.to_string(index=False)])
    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = calculate_errors()
    split = best_split_threshold(df)
    dominance = dominance_threshold(df)
    crossing = linear_crossing_threshold(df)
    region_summary = summarize_by_price_region(df)

    df.to_csv(OUTPUT_CSV, index=False)
    plot_path = plot_threshold_analysis(df, split, dominance, crossing)
    write_summary(split, dominance, crossing, region_summary)

    print(df)
    print()
    print(f"Best split threshold: C_el >= {split['threshold']:.2f} EUR/MWh -> use 2D")
    print(f"Correct decisions: {split['correct']} / {len(df)} ({split['accuracy']:.0%})")
    if dominance is not None:
        print(f"Strict observed dominance threshold: C_el >= {dominance:.2f} EUR/MWh")
    else:
        print("No strict observed dominance threshold found.")
    if crossing is not None:
        print(f"Linear trend crossing: C_el ~= {crossing:.2f} EUR/MWh")
    print(f"Saved point-wise CSV to: {OUTPUT_CSV}")
    print(f"Saved summary to: {OUTPUT_TXT}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
