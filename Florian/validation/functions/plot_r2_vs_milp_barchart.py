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
OUTPUT_PATH = VALIDATION_DIR / "r2_vs_milp_1d_vs_2d_sample_size_40_barchart.png"
R2_1D_PATH = VALIDATION_DIR / "results_1d_models" / "r2_vs_milp_summary.csv"
R2_2D_PATH = VALIDATION_DIR / "results_2d_models" / "r2_vs_milp_summary.csv"

SAMPLE_SIZE = 40
MODELS = ["MILP", "LP upper", "LP lower", "LP approx"]
MODEL_LABELS = {
    "MILP": "LR MILP",
    "LP upper": "LR LPupper",
    "LP lower": "LR LPlower",
    "LP approx": "LR LPapprox",
}
MODEL_COLORS = {
    "MILP": "tab:blue",
    "LP upper": "tab:orange",
    "LP lower": "tab:green",
    "LP approx": "tab:red",
}
SAMPLING_METHODS = ["1D", "2D"]
SAMPLING_HATCHES = {"1D": "", "2D": "///"}


def read_sample(path: Path, sampling_method: str) -> pd.Series:
    df = pd.read_csv(path)
    required = ["training_size", *MODELS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    sample = df.loc[df["training_size"] == SAMPLE_SIZE]
    if len(sample) != 1:
        raise ValueError(
            f"Expected exactly one row for sample size {SAMPLE_SIZE} in "
            f"{sampling_method} data, found {len(sample)}"
        )
    return sample.iloc[0][MODELS].astype(float)


def load_r2_values() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "1D": read_sample(R2_1D_PATH, "1D"),
            "2D": read_sample(R2_2D_PATH, "2D"),
        }
    ).reindex(MODELS)


def plot_r2_values(values: pd.DataFrame) -> Path:
    reset_plot_style()
    apply_style(width_cm=16, aspect=(4, 3), science=True, grid=False, latex=True)
    fig, ax = plt.subplots(figsize=get_figsize(width_cm=16, aspect=(4, 3)))
    x = np.arange(len(MODELS))
    width = 0.35

    for sampling_index, sampling_method in enumerate(SAMPLING_METHODS):
        offset = (sampling_index - 0.5) * width
        for model_index, model in enumerate(MODELS):
            ax.bar(
                x[model_index] + offset,
                values.loc[model, sampling_method],
                width=width,
                color=MODEL_COLORS[model],
                hatch=SAMPLING_HATCHES[sampling_method],
                edgecolor="black",
            )

    ax.set_title(
        f"Model prediction vs. MILP $R^2$ for 1D and 2D sampling "
        f"(sample size {SAMPLE_SIZE})"
    )
    ax.set_xlabel("Regression model")
    ax.set_ylabel("$R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[model] for model in MODELS], rotation=20, ha="right"
    )
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis="y", alpha=0.3)

    model_handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[model], ec="black")
        for model in MODELS
    ]
    fig.legend(
        model_handles,
        [MODEL_LABELS[model] for model in MODELS],
        title="Model",
        loc="lower center",
        ncol=len(MODELS),
        bbox_to_anchor=(0.5, 0.08),
    )
    sampling_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="black",
            hatch=SAMPLING_HATCHES[method],
        )
        for method in SAMPLING_METHODS
    ]
    fig.legend(
        sampling_handles,
        [f"{method} sampling" for method in SAMPLING_METHODS],
        title="Sampling",
        loc="lower center",
        ncol=len(SAMPLING_METHODS),
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.22, 1, 0.93))
    safe_figure(
        fig,
        save_path=OUTPUT_PATH.parent,
        filename=OUTPUT_PATH.stem,
        file_type=OUTPUT_PATH.suffix.lstrip("."),
    )
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    output = plot_r2_values(load_r2_values())
    print(f"Saved R2 vs. MILP bar chart to: {output}")
