from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
VALIDATION_DIR = ROOT / "Florian" / "validation"
OUTPUT_PATH = VALIDATION_DIR / "r2_by_sample_size_2d_milp_lpupper.png"
R2_2D_PATH = VALIDATION_DIR / "results_2d_models" / "intra_model_r2_scores_summary.csv"

SAMPLE_SIZES = [5, 20, 40]
MODELS = ["MILP", "LP upper"]
MODEL_LABELS = {
    "MILP": "LR MILP (2D)",
    "LP upper": "LR LPupper (2D)",
}
MODEL_COLORS = {
    "MILP": "tab:blue",
    "LP upper": "tab:orange",
}


def reset_plot_settings() -> None:
    plt.rcdefaults()


def load_r2_by_sample_size() -> pd.DataFrame:
    if not R2_2D_PATH.exists():
        raise FileNotFoundError(f"Missing 2D R2 summary file: {R2_2D_PATH}")

    df = pd.read_csv(R2_2D_PATH)
    required_columns = ["training_size", *MODELS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {R2_2D_PATH}: {missing_columns}")

    df = df[df["training_size"].isin(SAMPLE_SIZES)]
    missing_sample_sizes = sorted(set(SAMPLE_SIZES) - set(df["training_size"]))
    if missing_sample_sizes:
        raise ValueError(f"Missing sample sizes in {R2_2D_PATH}: {missing_sample_sizes}")

    return df.sort_values("training_size")


def plot_r2_by_sample_size() -> Path:
    regression_r2 = load_r2_by_sample_size()

    reset_plot_settings()
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for model in MODELS:
        ax.plot(
            regression_r2["training_size"],
            regression_r2[model],
            marker="o",
            linewidth=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )

    ax.set_title("2D regression model $R^2$ by sample size")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("$R^2$")
    ax.set_xticks(SAMPLE_SIZES)
    ax.set_ylim(0.9, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)

    print(f"Saved 2D R2 by sample size plot to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    plot_r2_by_sample_size()
