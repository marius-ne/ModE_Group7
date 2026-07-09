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
OUTPUT_PATH = VALIDATION_DIR / "regression_r2_by_sample_size_barcharts.png"

SAMPLE_SIZES = [40, 20, 5]
MODELS = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]

MODEL_LABELS = {
    "opex_milp": "MILP",
    "opex_lp_lower": "LP Lower",
    "opex_lp_upper": "LP Upper",
    "opex_lp_approx": "LP Approx",
}

SUMMARY_MODEL_NAMES = {
    "opex_milp": "MILP_pred",
    "opex_lp_lower": "LP_Lower",
    "opex_lp_upper": "LP_Upper",
    "opex_lp_approx": "LP_Approx",
}

COLORS = {
    "opex_milp": "tab:blue",
    "opex_lp_lower": "tab:orange",
    "opex_lp_upper": "tab:green",
    "opex_lp_approx": "tab:red",
}


def inter_model_csv_path(sample_size: int, model: str) -> Path:
    return VALIDATION_DIR / f"{sample_size}_train_10_test_ratio_{model}.csv"


def milp_summary_csv_path(sample_size: int) -> Path:
    patterns = [
        f"{sample_size}_train_10_test_r2_score_compared_to_milp.csv",
        f"{sample_size}_train_10_predict_r2_score_compared_to_milp.csv",
    ]
    for pattern in patterns:
        path = VALIDATION_DIR / pattern
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing R2-vs-MILP summary for sample size {sample_size}. "
        f"Expected one of: {patterns}"
    )


def load_inter_model_r2() -> pd.DataFrame:
    rows = []
    for sample_size in SAMPLE_SIZES:
        for model in MODELS:
            path = inter_model_csv_path(sample_size, model)
            if not path.exists():
                raise FileNotFoundError(f"Missing inter-model R2 file: {path}")

            df = pd.read_csv(path)
            rows.append(
                {
                    "sample_size": sample_size,
                    "model": model,
                    "r2": float(df.iloc[0, -1]),
                }
            )

    return pd.DataFrame(rows)


def load_milp_comparison_r2() -> pd.DataFrame:
    rows = []
    for sample_size in SAMPLE_SIZES:
        path = milp_summary_csv_path(sample_size)
        df = pd.read_csv(path)

        for model in MODELS:
            summary_name = SUMMARY_MODEL_NAMES[model]
            match = df[df["model"] == summary_name]
            if match.empty:
                raise ValueError(f"Missing model '{summary_name}' in {path}")

            rows.append(
                {
                    "sample_size": sample_size,
                    "model": model,
                    "r2": float(match["r2_compared_to_actual_milp_opex"].iloc[0]),
                }
            )

    return pd.DataFrame(rows)


def plot_grouped_bars(ax, df: pd.DataFrame, title: str) -> None:
    x = np.arange(len(SAMPLE_SIZES))
    width = 0.18
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * width

    for offset, model in zip(offsets, MODELS):
        values = [
            df[(df["sample_size"] == sample_size) & (df["model"] == model)]["r2"].iloc[0]
            for sample_size in SAMPLE_SIZES
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )

    y_min = df["r2"].min()
    y_max = df["r2"].max()
    y_padding = max((y_max - y_min) * 0.2, 0.01)

    ax.set_title(title)
    ax.set_ylabel("$R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(sample_size) for sample_size in SAMPLE_SIZES])
    ax.set_ylim(max(0.0, y_min - y_padding), min(1.0, y_max + y_padding))
    ax.grid(axis="y", alpha=0.3)


def plot_r2_barcharts() -> Path:
    inter_model_r2 = load_inter_model_r2()
    milp_comparison_r2 = load_milp_comparison_r2()

    reset_plot_settings()
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=get_figsize(16, (16, 12)),
        sharex=True,
    )

    plot_grouped_bars(
        axes[0],
        inter_model_r2,
        "Inter-model R2 by training sample size",
    )
    plot_grouped_bars(
        axes[1],
        milp_comparison_r2,
        "R2 compared to MILP OPEX by training sample size",
    )

    axes[1].set_xlabel("Training sample size")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", loc="upper center", ncol=len(MODELS))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)

    print(f"Saved R2 bar charts to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    plot_r2_barcharts()
