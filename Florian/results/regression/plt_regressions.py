from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Erdem"))

from src.visualization.style import get_figsize, reset_plot_settings


RESULTS_DIR = Path("Florian/results/regression")
TEST_DATA_PATH = Path("Marius/results/opex_random_sample_10.csv")

MODEL_PATHS = {
    "opex_milp": Path("Florian/surrogate_model_opex_milp.joblib"),
    "opex_lp_lower": Path("Florian/surrogate_model_opex_lp_lower.joblib"),
    "opex_lp_upper": Path("Florian/surrogate_model_opex_lp_upper.joblib"),
    "opex_lp_approx": Path("Florian/surrogate_model_opex_lp_approx.joblib"),
}

VALIDATION_FILES = {
    "opex_milp": RESULTS_DIR / "validation_opex_milp.csv",
    "opex_lp_lower": RESULTS_DIR / "validation_opex_lp_lower.csv",
    "opex_lp_upper": RESULTS_DIR / "validation_opex_lp_upper.csv",
    "opex_lp_approx": RESULTS_DIR / "validation_opex_lp_approx.csv",
}

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
    "MILP": "MILP",
    "LP_Lower": "LP Lower",
    "LP_Upper": "LP Upper",
    "LP_Approx": "LP Approx",
}

TEST_TARGET_COLUMNS = {
    "opex_milp": "opex_milp",
    "opex_lp_lower": "opex_lp_lower",
    "opex_lp_upper": "opex_lp_upper",
    "opex_lp_approx": "opex_lp_approximated",
}


def plot_regression_lines_with_test_data():
    df_test = pd.read_csv(TEST_DATA_PATH)
    x_test = df_test[["ratio"]]
    x_line_values = np.linspace(df_test["ratio"].min(), df_test["ratio"].max(), 200)
    x_line = pd.DataFrame({"ratio": x_line_values})
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

    reset_plot_settings()
    plt.figure(figsize=get_figsize(16, "golden"))

    for model_name, model in models.items():
        y_line = model.predict(x_line)
        intercept = model.intercept_
        slope = model.coef_[0]
        target_column = TEST_TARGET_COLUMNS[model_name]
        y_actual = df_test[target_column]
        model_label = MODEL_LABELS[model_name]

        plt.plot(
            x_line_values,
            y_line,
            color=COLORS[model_name],
            linewidth=2,
            label=f"{model_label}: y = {slope:.2f}x + {intercept:.2f}",
        )

        plt.scatter(
            df_test["ratio"],
            y_actual,
            color=COLORS[model_name],
            edgecolors="black",
            alpha=0.75,
            linewidths=0.5,
            label=f"{model_label} test data",
        )

    plt.xlabel("Ratio")
    plt.ylabel("OPEX/C_el")
    plt.title("Regression Lines with Test Data")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("Florian/results/regression/regression_lines.png", dpi=300)
    plt.close()

def plot_single_regression_lines_with_data(model_name):
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of: {list(MODEL_PATHS)}"
        )

    df_test = pd.read_csv(TEST_DATA_PATH)
    x_line_values = np.linspace(df_test["ratio"].min(), df_test["ratio"].max(), 200)
    x_line = pd.DataFrame({"ratio": x_line_values})
    model = joblib.load(MODEL_PATHS[model_name])
    model_label = MODEL_LABELS[model_name]

    reset_plot_settings()
    plt.figure(figsize=get_figsize(16, "golden"))

    y_line = model.predict(x_line)
    intercept = model.intercept_
    slope = model.coef_[0]
    target_column = TEST_TARGET_COLUMNS[model_name]
    y_actual = df_test[target_column]

    plt.plot(
            x_line_values,
            y_line,
            color=COLORS[model_name],
            linewidth=2,
            label=f"{model_label}: y = {slope:.2f}x + {intercept:.2f}",
        )

    plt.scatter(
        df_test["ratio"],
        y_actual,
        color=COLORS[model_name],
        edgecolors="black",
        alpha=0.75,
        linewidths=0.5,
        label=f"{model_label} test data",
        )
    plt.xlabel("Ratio")
    plt.ylabel("OPEX/C_el")
    plt.title(f"Regression Line with Test Data ({model_label})")
    plt.grid(True)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / f"regression_line_{model_name}.png", dpi=300)
    plt.close()


def plot_single_model(target, validation_df):
    y_test = validation_df["y_test"]
    y_pred = validation_df["y_pred"]

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    reset_plot_settings()
    plt.figure(figsize=get_figsize(16, "golden"))
    plt.scatter(y_test, y_pred, color=COLORS[target], edgecolors="black", alpha=0.7)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        lw=2,
        label="Ideal prediction",
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(
        f"Actual vs. Predicted OPEX/C_el ({MODEL_LABELS[target]}); "
        f"R^2: {r2:.4f}; MSE: {mse:,.2f}"
    )
    plt.legend()
    plt.grid(True)
    plt.xlim(min_val * 0.9, max_val * 1.05)
    plt.ylim(min_val * 0.9, max_val * 1.05)

    plot_path = RESULTS_DIR / f"actual_vs_predicted_{target}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")


def plot_all_models(combined_df):
    reset_plot_settings()
    plt.figure(figsize=get_figsize(16, "golden"))

    for target in VALIDATION_FILES:
        target_df = combined_df[combined_df["target"] == target]
        plt.scatter(
            target_df["actual"],
            target_df["predicted"],
            color=COLORS[target],
            alpha=0.65,
            edgecolors="black",
            linewidths=0.4,
            label=MODEL_LABELS[target],
        )

    min_val = min(combined_df["actual"].min(), combined_df["predicted"].min())
    max_val = max(combined_df["actual"].max(), combined_df["predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Ideal prediction")

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted OPEX/C_el - All Models")
    plt.legend(title="Model")
    plt.grid(True)
    plt.xlim(min_val * 0.9, max_val * 1.05)
    plt.ylim(min_val * 0.9, max_val * 1.05)

    combined_plot_path = RESULTS_DIR / "actual_vs_predicted_all_models.png"
    plt.savefig(combined_plot_path, dpi=300)
    plt.close()
    print(f"Combined plot saved to: {combined_plot_path}")


def plot_r2_scores_by_training_size(score_type):
    score_configs = {
        "compared": {
            "files": {
                5: RESULTS_DIR / "r2_score_5_compared.csv",
                20: RESULTS_DIR / "r2_score_20_compared.csv",
                40: RESULTS_DIR / "r2_score_compared.csv",
            },
            "r2_column": "r2_actual_vs_predicted",
            "title": "R^2 by Training Sample Size (Model Actual vs. Model Prediction)",
            "output": RESULTS_DIR / "r2_by_training_size_compared.png",
        },
        "compared_to_actual_milp_opex": {
            "files": {
                5: RESULTS_DIR / "r2_score_5_compared_to_actual_milp_opex.csv",
                20: RESULTS_DIR / "r2_score_20_compared_to_actual_milp_opex.csv",
                40: RESULTS_DIR / "r2_score_compared_to_actual_milp_opex.csv",
            },
            "r2_column": "r2_compared_to_actual_milp_opex",
            "title": "R^2 by Training Sample Size (MILP Actual vs. Model Prediction)",
            "output": RESULTS_DIR / "r2_by_training_size_compared_to_actual_milp_opex.png",
        },
    }
    score_type_aliases = {
        "actual_milp": "compared_to_actual_milp_opex",
    }
    score_type = score_type_aliases.get(score_type, score_type)
    if score_type not in score_configs:
        raise ValueError(f"score_type must be one of {list(score_configs)}.")

    config = score_configs[score_type]
    model_to_color_key = {
        "MILP": "opex_milp",
        "LP_Lower": "opex_lp_lower",
        "LP_Upper": "opex_lp_upper",
        "LP_Approx": "opex_lp_approx",
    }
    model_order = ["MILP", "LP_Upper", "LP_Lower", "LP_Approx"]

    rows = []
    for training_size, path in config["files"].items():
        df_scores = pd.read_csv(path)
        for _, row in df_scores.iterrows():
            model = "MILP" if row["model"] == "MILP_pred" else row["model"]
            if model not in model_to_color_key:
                continue

            rows.append(
                {
                    "training_size": training_size,
                    "model": model,
                    "r2": row[config["r2_column"]],
                }
            )

    df_plot = pd.DataFrame(rows)

    reset_plot_settings()
    plt.figure(figsize=get_figsize(16, "golden"))
    for model in model_order:
        model_df = df_plot[df_plot["model"] == model].sort_values("training_size")
        if model_df.empty:
            continue

        color_key = model_to_color_key[model]
        plt.plot(
            model_df["training_size"],
            model_df["r2"],
            color=COLORS[color_key],
            linewidth=2,
            marker="o",
            markersize=6,
            label=MODEL_LABELS[model],
        )

    y_min = df_plot["r2"].min()
    y_max = df_plot["r2"].max()
    y_padding = max((y_max - y_min) * 0.15, 0.001)

    plt.xlabel("Training sample size")
    plt.ylabel("R^2 Score")
    plt.title(config["title"])
    plt.xticks([5, 20, 40])
    plt.xlim(3, 42)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    plt.grid(True)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(config["output"], dpi=300)
    plt.close()
    print(f"R2 plot saved to: {config['output']}")


def regenerate_all_regression_plots():
    combined_predictions = []

    plot_regression_lines_with_test_data()

    for target, validation_path in VALIDATION_FILES.items():
        validation_df = pd.read_csv(validation_path)
        plot_single_model(target, validation_df)

        combined_predictions.append(
            pd.DataFrame(
                {
                    "target": target,
                    "actual": validation_df["y_test"].to_numpy(),
                    "predicted": validation_df["y_pred"].to_numpy(),
                }
            )
        )

    combined_df = pd.concat(combined_predictions, ignore_index=True)
    plot_all_models(combined_df)

    for model_name in MODEL_PATHS:
        plot_single_regression_lines_with_data(model_name)

    plot_r2_scores_by_training_size("compared")
    plot_r2_scores_by_training_size("compared_to_actual_milp_opex")


def main():
    regenerate_all_regression_plots()


if __name__ == "__main__":
    main()
