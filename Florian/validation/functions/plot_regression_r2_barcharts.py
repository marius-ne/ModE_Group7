from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ERDEM_DIR = ROOT / "Erdem"
if str(ERDEM_DIR) not in sys.path:
    sys.path.insert(0, str(ERDEM_DIR))

from src.visualization.style import (  # noqa: E402
    apply_style,
    get_figsize,
    reset_plot_style,
    safe_figure,
)

VALIDATION_DIR = ROOT / "Florian" / "validation"
OUTPUT_PATH = VALIDATION_DIR / "regression_r2_1d_vs_2d_sample_size_40_barchart.png"
R2_1D_PATH = VALIDATION_DIR / "results_1d_models" / "intra_model_absolute_opex_r2_by_training_size.csv"
R2_2D_PATH = VALIDATION_DIR / "results_2d_models" / "intra_model_r2_scores_summary.csv"

SAMPLE_SIZE = 40
MODELS = ["opex_milp", "opex_lp_upper", "opex_lp_lower", "opex_lp_approx"]
SAMPLING_METHODS = ["1D", "2D"]

MODEL_LABELS = {
    "opex_milp": "LR MILP",
    "opex_lp_upper": "LR LPupper",
    "opex_lp_lower": "LR LPlower",
    "opex_lp_approx": "LR LPapprox",
}

CSV_MODEL_COLUMNS = {
    "opex_milp": "MILP",
    "opex_lp_upper": "LP upper",
    "opex_lp_lower": "LP lower",
    "opex_lp_approx": "LP approx",
}

MODEL_COLORS = {
    "opex_milp": "tab:blue",
    "opex_lp_upper": "tab:orange",
    "opex_lp_lower": "tab:green",
    "opex_lp_approx": "tab:red",
}

SAMPLING_HATCHES = {
    "1D": "",
    "2D": "///",
}


def read_summary(path: Path, sampling_method: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {sampling_method} R2 summary file: {path}")

    df = pd.read_csv(path)
    required_columns = ["training_size", *CSV_MODEL_COLUMNS.values()]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")

    return df.set_index("training_size")


def load_regression_r2() -> pd.DataFrame:
    r2_1d = read_summary(R2_1D_PATH, "1D")
    r2_2d = read_summary(R2_2D_PATH, "2D")
    if SAMPLE_SIZE not in r2_1d.index:
        raise ValueError(f"Missing sample size {SAMPLE_SIZE} in {R2_1D_PATH}")
    if SAMPLE_SIZE not in r2_2d.index:
        raise ValueError(f"Missing sample size {SAMPLE_SIZE} in {R2_2D_PATH}")

    rows = []
    for model, csv_column in CSV_MODEL_COLUMNS.items():
        rows.extend(
            [
                {
                    "model": model,
                    "sampling_method": "1D",
                    "r2": float(r2_1d.loc[SAMPLE_SIZE, csv_column]),
                },
                {
                    "model": model,
                    "sampling_method": "2D",
                    "r2": float(r2_2d.loc[SAMPLE_SIZE, csv_column]),
                },
            ]
        )

    result = pd.DataFrame(rows)
    return (
        result
        .pivot(index="model", columns="sampling_method", values="r2")
        .reindex(MODELS)[SAMPLING_METHODS]
    )


def plot_grouped_bars(ax, df: pd.DataFrame) -> None:
    x = np.arange(len(df.index))
    width = 0.35
    offsets = (np.arange(len(df.columns)) - (len(df.columns) - 1) / 2) * width

    for offset, sampling_method in zip(offsets, df.columns):
        for idx, model in enumerate(df.index):
            value = df[sampling_method].loc[model]
            ax.bar(
                x[idx] + offset,
                value,
                width=width,
                color=MODEL_COLORS[model],
                hatch=SAMPLING_HATCHES[sampling_method],
                edgecolor="black",
            )

    ax.set_title(f"Regression model $R^2$ for 1D and 2D sampling (sample size {SAMPLE_SIZE})")
    ax.set_ylabel("$R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in df.index], rotation=20, ha="right")
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis="y", alpha=0.3)


def add_legends(fig) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[model], ec="black")
        for model in MODELS
    ]
    labels = [MODEL_LABELS[model] for model in MODELS]
    model_legend = fig.legend(handles, labels, title="Model", loc="lower center", ncol=len(MODELS), bbox_to_anchor=(0.5, 0.08))

    hatch_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=SAMPLING_HATCHES[method])
        for method in SAMPLING_METHODS
    ]
    hatch_labels = [f"{method} sampling" for method in SAMPLING_METHODS]
    fig.legend(hatch_handles, hatch_labels, title="Sampling", loc="lower center", ncol=len(SAMPLING_METHODS), bbox_to_anchor=(0.5, 0.0))
    return model_legend


def plot_r2_barcharts() -> Path:
    regression_r2 = load_regression_r2()

    # Apply the shared plotting conventions before creating any figure or axes.
    reset_plot_style()
    apply_style(
        width_cm=16,
        aspect=(16, 12),
        nrows=1,
        ncols=1,
        science=True,
        grid=False,
        latex=True,
    )
    figure_size = get_figsize(width_cm=16, aspect=(16, 12), nrows=1, ncols=1)
    fig, ax = plt.subplots(figsize=figure_size)

    plot_grouped_bars(ax, regression_r2)
    ax.set_xlabel("Regression model")
    add_legends(fig)
    fig.tight_layout(rect=(0, 0.22, 1, 0.93))

    safe_figure(
        fig,
        save_path=OUTPUT_PATH.parent,
        filename=OUTPUT_PATH.stem,
        file_type=OUTPUT_PATH.suffix.lstrip("."),
    )
    plt.close(fig)

    print(f"Saved R2 bar charts to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    plot_r2_barcharts()
