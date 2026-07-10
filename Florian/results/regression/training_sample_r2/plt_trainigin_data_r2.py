from pathlib import Path
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.visualization.style import get_figsize, reset_plot_settings


DATA_DIR = ROOT / "Florian" / "results" / "regression" / "training_sample_r2"
OUTPUT_PATH = DATA_DIR / "training_sample_r2_by_size.png"

TRAINING_SAMPLE_SIZES = [5, 20, 40]
MODELS = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]

COLORS = {
    "opex_milp": "tab:blue",
    "opex_lp_lower": "tab:orange",
    "opex_lp_upper": "tab:green",
    "opex_lp_approx": "tab:red",
}

MODEL_LABELS = {
    "opex_milp": "MILP",
    "opex_lp_lower": "LP Lower",
    "opex_lp_upper": "LP Upper",
    "opex_lp_approx": "LP Approx",
}


def load_training_r2_values() -> pd.DataFrame:
    rows = []

    for training_size in TRAINING_SAMPLE_SIZES:
        for model in MODELS:
            csv_path = DATA_DIR / f"training_data_r2_mse_{training_size}_{model}.csv"
            if not csv_path.exists():
                print(f"Skipping missing file: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            rows.append(
                {
                    "training_sample_size": training_size,
                    "model": model,
                    "r2_score_total": float(df["r2_score_total"].iloc[0]),
                }
            )

    if not rows:
        raise FileNotFoundError(f"No training R2 CSV files found in {DATA_DIR}")

    return pd.DataFrame(rows)


def plot_training_r2_by_sample_size():
    df_plot = load_training_r2_values()

    reset_plot_settings()
    plt.figure(figsize=get_figsize(16, "golden"))

    for model in MODELS:
        model_df = df_plot[df_plot["model"] == model].sort_values("training_sample_size")
        if model_df.empty:
            continue

        plt.plot(
            model_df["training_sample_size"],
            model_df["r2_score_total"],
            color=COLORS[model],
            linewidth=2,
            marker="o",
            markersize=6,
            label=MODEL_LABELS[model],
        )

    y_min = df_plot["r2_score_total"].min()
    y_max = df_plot["r2_score_total"].max()
    y_padding = max((y_max - y_min) * 0.15, 0.001)

    plt.xlabel("Training sample size")
    plt.ylabel("R^2 score on training data")
    plt.title("Training Data R^2 by Training Sample Size")
    plt.xticks(TRAINING_SAMPLE_SIZES)
    plt.xlim(min(TRAINING_SAMPLE_SIZES) - 2, max(TRAINING_SAMPLE_SIZES) + 2)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    plt.grid(True)
    plt.legend(title="Model")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()
    print(f"Training R2 plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_training_r2_by_sample_size()
