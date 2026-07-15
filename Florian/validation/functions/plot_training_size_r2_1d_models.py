from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.visualization.style import apply_style, safe_figure


from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
VALIDATION_DIR = ROOT / "Florian" / "validation"
OUTPUT_PATH = VALIDATION_DIR / "training_size_r2_1d_milp_lpupper.png"

TRAINING_SIZES = [5, 20, 40]
MODELS = ["opex_milp", "opex_lp_upper"]
MODEL_LABELS = {
    "opex_milp": "MILP",
    "opex_lp_upper": "LP upper",
}
MODEL_COLORS = {
    "opex_milp": "tab:blue",
    "opex_lp_upper": "tab:orange",
}


def reset_plot_settings():
    plt.rcdefaults()


def load_r2_by_training_size() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for training_size in TRAINING_SIZES:
            path = VALIDATION_DIR / "results_1d_models" / f"{training_size}_train_10_test_ratio_{model}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing regression results file: {path}")

            df = pd.read_csv(path)
            if "r2" not in df.columns:
                raise ValueError(f"Missing 'r2' column in {path}")

            rows.append(
                {
                    "training_size": training_size,
                    "model": model,
                    "r2": float(df["r2"].iloc[0]),
                }
            )

    result = pd.DataFrame(rows)
    pivot = result.pivot(index="training_size", columns="model", values="r2")
    pivot = pivot.reindex(TRAINING_SIZES)
    return pivot


def plot_r2_vs_training_size() -> Path:
    regression_r2 = load_r2_by_training_size()

    reset_plot_settings()
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for model in MODELS:
        ax.plot(
            regression_r2.index,
            regression_r2[model],
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )

    ax.set_title("1D model $R^2$ by training sample size")
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("$R^2$")
    ax.set_xticks(TRAINING_SIZES)
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)

    print(f"Saved training-size R2 plot to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    plot_r2_vs_training_size()
