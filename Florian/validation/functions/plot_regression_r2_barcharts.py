from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]

def reset_plot_settings():
    plt.rcdefaults()

def get_figsize(width, ratio):
    return (10, 7.5)

VALIDATION_DIR = ROOT / "Florian" / "validation"
OUTPUT_PATH = VALIDATION_DIR / "regression_r2_1d_vs_2d_sample_size_40_barchart.png"

SAMPLE_SIZE = 40
MODELS = ["opex_milp", "opex_lp_upper", "opex_lp_lower", "opex_lp_approx"]
SAMPLING_METHODS = ["1D", "2D"]

MODEL_LABELS = {
    "opex_milp": "LR MILP",
    "opex_lp_upper": "LR LPupper",
    "opex_lp_lower": "LR LPlower",
    "opex_lp_approx": "LR LPapprox",
}

COLORS = {
    "1D": "tab:blue",
    "2D": "tab:orange",
}


def regression_csv_path(sample_size: int, model: str, sampling_method: str) -> Path:
    if sampling_method == "1D":
        return VALIDATION_DIR / "results_1d_models" / f"{sample_size}_train_10_test_ratio_{model}.csv"
    if sampling_method == "2D":
        return VALIDATION_DIR / "results_2d_models" / f"{sample_size}_train_10_test_2d_discrete_{model}.csv"
    raise ValueError(f"Unknown sampling method: {sampling_method}")


def load_regression_r2() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for sampling_method in SAMPLING_METHODS:
            path = regression_csv_path(SAMPLE_SIZE, model, sampling_method)
            if not path.exists():
                raise FileNotFoundError(f"Missing regression R2 file: {path}")

            df = pd.read_csv(path)
            if "r2" not in df.columns:
                raise ValueError(f"Missing 'r2' column in {path}")

            rows.append(
                {
                    "model": model,
                    "sampling_method": sampling_method,
                    "r2": float(df["r2"].iloc[0]),
                }
            )

    return pd.DataFrame(rows)


def plot_grouped_bars(ax, df: pd.DataFrame) -> None:
    x = np.arange(len(MODELS))
    width = 0.35
    offsets = (np.arange(len(SAMPLING_METHODS)) - (len(SAMPLING_METHODS) - 1) / 2) * width

    for offset, sampling_method in zip(offsets, SAMPLING_METHODS):
        values = [
            df[(df["model"] == model) & (df["sampling_method"] == sampling_method)]["r2"].iloc[0]
            for model in MODELS
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[sampling_method],
            label=sampling_method,
        )

    y_min = df["r2"].min()
    y_max = df["r2"].max()
    y_padding = max((y_max - y_min) * 0.2, 0.01)

    ax.set_title(f"Regression model $R^2$ for 1D and 2D sampling (sample size {SAMPLE_SIZE})")
    ax.set_ylabel("$R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODELS])
    ax.set_ylim(max(0.0, y_min - y_padding), min(1.0, y_max + y_padding))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Sampling method")


def plot_r2_barcharts() -> Path:
    regression_r2 = load_regression_r2()

    reset_plot_settings()
    fig, ax = plt.subplots(
        figsize=get_figsize(16, (16, 12)),
    )

    plot_grouped_bars(ax, regression_r2)
    ax.set_xlabel("Regression model")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)

    print(f"Saved R2 bar charts to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    plot_r2_barcharts()
