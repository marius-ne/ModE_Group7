from pathlib import Path
import re
import sys

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.visualization.style import get_figsize, reset_plot_settings


OUTPUT_DIR = Path(__file__).resolve().parent
MARIUS_RESULTS_DIR = ROOT / "Marius" / "results"
JOBLIB_DIR = ROOT / "Florian" / "surrogate_models" / "joblibs"

OUTPUT_CSV = OUTPUT_DIR / "r2_score_compared_to_milp_training_data.csv"
OUTPUT_PLOT = OUTPUT_DIR / "r2_by_training_size_compared_to_milp_training_data.png"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions_compared_to_milp_training_data.csv"

MODEL_PATHS = {
    "MILP_pred": JOBLIB_DIR / "surrogate_model_ratio_opex_milp.joblib",
    "LP_Upper": JOBLIB_DIR / "surrogate_model_ratio_opex_lp_upper.joblib",
    "LP_Lower": JOBLIB_DIR / "surrogate_model_ratio_opex_lp_lower.joblib",
    "LP_Approx": JOBLIB_DIR / "surrogate_model_ratio_opex_lp_approx.joblib",
}

MODEL_ORDER = ["MILP_pred", "LP_Upper", "LP_Lower", "LP_Approx"]

MODEL_LABELS = {
    "MILP_pred": "MILP",
    "LP_Upper": "LP Upper",
    "LP_Lower": "LP Lower",
    "LP_Approx": "LP Approx",
}

COLORS = {
    "MILP_pred": "tab:blue",
    "LP_Upper": "tab:green",
    "LP_Lower": "tab:orange",
    "LP_Approx": "tab:red",
}


def get_training_sample_size(path: Path) -> int:
    match = re.search(r"evaluation_log_samples_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Could not extract training sample size from {path.name}")
    return int(match.group(1))


def find_training_sample_files() -> list[tuple[int, Path]]:
    files = sorted(
        MARIUS_RESULTS_DIR.glob("evaluation_log_samples_*.csv"),
        key=get_training_sample_size,
    )
    if not files:
        raise FileNotFoundError(
            f"No evaluation_log_samples_*.csv files found in {MARIUS_RESULTS_DIR}"
        )
    return [(get_training_sample_size(path), path) for path in files]


def load_models():
    missing_models = [path for path in MODEL_PATHS.values() if not path.exists()]
    if missing_models:
        missing = "\n".join(str(path) for path in missing_models)
        raise FileNotFoundError(f"Missing joblib model files:\n{missing}")

    return {model_name: joblib.load(path) for model_name, path in MODEL_PATHS.items()}


def calculate_r2_scores():
    models = load_models()
    score_rows = []
    prediction_rows = []

    for training_size, sample_path in find_training_sample_files():
        df_samples = pd.read_csv(sample_path)
        required_columns = {"ratio", "opex_milp"}
        missing_columns = required_columns - set(df_samples.columns)
        if missing_columns:
            raise ValueError(
                f"{sample_path} is missing required columns: {sorted(missing_columns)}"
            )

        x_ratio = df_samples[["ratio"]]
        y_milp_training = df_samples["opex_milp"]

        for model_name in MODEL_ORDER:
            y_pred = models[model_name].predict(x_ratio)
            r2 = r2_score(y_milp_training, y_pred)

            score_rows.append(
                {
                    "training_sample_size": training_size,
                    "model": model_name,
                    "r2_compared_to_milp_training_data": r2,
                }
            )

            for ratio, y_true, prediction in zip(
                df_samples["ratio"], y_milp_training, y_pred
            ):
                prediction_rows.append(
                    {
                        "training_sample_size": training_size,
                        "model": model_name,
                        "ratio": ratio,
                        "opex_milp_training_data": y_true,
                        "y_pred": prediction,
                    }
                )

    df_scores = pd.DataFrame(score_rows)
    df_predictions = pd.DataFrame(prediction_rows)

    milp_scores = df_scores[df_scores["model"] == "MILP_pred"][
        ["training_sample_size", "r2_compared_to_milp_training_data"]
    ].rename(columns={"r2_compared_to_milp_training_data": "milp_pred_r2"})
    df_scores = df_scores.merge(milp_scores, on="training_sample_size", how="left")
    df_scores["delta_to_milp_pred"] = (
        df_scores["r2_compared_to_milp_training_data"] - df_scores["milp_pred_r2"]
    )
    df_scores = df_scores.drop(columns="milp_pred_r2")

    return df_scores, df_predictions


def plot_r2_scores(df_scores: pd.DataFrame):
    reset_plot_settings()
    plt.figure(figsize=get_figsize(16, "golden"))

    for model_name in MODEL_ORDER:
        model_df = df_scores[df_scores["model"] == model_name].sort_values(
            "training_sample_size"
        )
        if model_df.empty:
            continue

        plt.plot(
            model_df["training_sample_size"],
            model_df["r2_compared_to_milp_training_data"],
            color=COLORS[model_name],
            linewidth=2,
            marker="o",
            markersize=6,
            label=MODEL_LABELS[model_name],
        )

    y_min = df_scores["r2_compared_to_milp_training_data"].min()
    y_max = df_scores["r2_compared_to_milp_training_data"].max()
    y_padding = max((y_max - y_min) * 0.15, 0.001)
    training_sizes = sorted(df_scores["training_sample_size"].unique())

    plt.xlabel("Training sample size")
    plt.ylabel("R^2 Score")
    plt.title("R^2 by Training Sample Size (MILP Training Data vs. Model Prediction)")
    plt.xticks(training_sizes)
    plt.xlim(min(training_sizes) - 2, max(training_sizes) + 2)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    plt.grid(True)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()


def main():
    df_scores, df_predictions = calculate_r2_scores()
    df_scores.to_csv(OUTPUT_CSV, index=False)
    df_predictions.to_csv(PREDICTIONS_CSV, index=False)
    plot_r2_scores(df_scores)

    print(df_scores)
    print(f"R2 CSV saved to: {OUTPUT_CSV}")
    print(f"Prediction details saved to: {PREDICTIONS_CSV}")
    print(f"R2 plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
