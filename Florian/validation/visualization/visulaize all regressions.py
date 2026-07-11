from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[3]
TRAINING_DATA_PATH = ROOT / "Marius" / "results" / "evaluation_training_samples_1D_angle.csv"
MODEL_DIR = ROOT / "Florian" / "validation" / "joblibs"
OUTPUT_PATH = Path(__file__).resolve().parent / "all_1d_regression_models_training_points.png"
TRAINING_SIZE = 40

TARGETS = [
    ("milp", "MILP", "tab:blue"),
    ("lp_lower", "LP lower", "tab:orange"),
    ("lp_upper", "LP upper", "tab:green"),
    ("lp_approx", "LP approx", "tab:red"),
]


def reset_plot_settings() -> None:
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(TRAINING_DATA_PATH)
    if "ratio" not in df.columns:
        raise ValueError(f"Expected 'ratio' column in {TRAINING_DATA_PATH}")
    return df


def plot_regression_lines() -> Path:
    reset_plot_settings()
    train_df = load_training_data()

    fig, ax = plt.subplots(figsize=(12, 7.5))
    x = train_df["ratio"].to_numpy(dtype=float)

    for target, label, color in TARGETS:
        y_true = train_df[f"opex_{target}"].to_numpy(dtype=float)
        model_path = MODEL_DIR / f"{TRAINING_SIZE}_ratio_opex_{target}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        y_pred = model.predict(x.reshape(-1, 1)).astype(float)
        r2 = r2_score(y_true, y_pred)

        order = np.argsort(x)
        ax.scatter(
            x,
            y_true,
            s=55,
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.plot(
            x[order],
            y_pred[order],
            color=color,
            linewidth=2.2,
            label=f"{label} (R² = {r2:.3f})",
            zorder=2,
        )

    ax.set_title("1D regression lines vs. 40 training points for all targets")
    ax.set_xlabel("ratio = gas price / electricity price")
    ax.set_ylabel("OPEX")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)

    print(f"Saved plot to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    plot_regression_lines()
