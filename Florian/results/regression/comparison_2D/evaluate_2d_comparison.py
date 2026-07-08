from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Erdem"))

from src.visualization.style import get_figsize, reset_plot_settings
from Florian.results.regression.plt_regressions import TEST_DATA_PATH, TRAIN_DATA_PATH


# Auswahlmoeglichkeit:
# "inter_model_r2" = bisherige Auswertung wie zuvor
# "compared_to_milp_r2" = alle y_pred gegen das echte MILP-OPEX aus Marius/results/opex_random_sample_10.csv
# "mse_vs_milp" = MILP-Testwerte mit y_pred eines waehlbaren Modells per MSE vergleichen
# "plot_train_and_test_data_for_all_models"
# "plot_3d_price_regressions" = 3D-Regressionsebenen fuer konkrete Gas- und Strompreise
RUN_MODE = "plot_train_and_test_data_for_all_models"  # "inter_model_r2", "compared_to_milp_r2", "mse_vs_milp", "plot_train_and_test_data_for_all_models", or "plot_3d_price_regressions"
MSE_MODEL = "MILP_pred"

RESULTS_DIR = Path("Florian/results/regression/comparison_2D")
MARIUS_RESULTS_DIR = Path("Marius/results")

TRAIN_DATA_PATH = Path("Marius/results/opex_specific_LHS_2D_sample_40_.csv")
TEST_DATA_PATH = Path("Marius/results/opex_random_sample_10.csv")

MODEL_PATHS = {
    "opex_milp": Path("Florian/surrogate_models/joblibs/surrogate_model_2d_sampling_1d_training_ratio_opex_milp.joblib"),
    "opex_lp_lower": Path("Florian/surrogate_models/joblibs/surrogate_model_2d_sampling_1d_training_ratio_opex_lp_lower.joblib"),
    "opex_lp_upper": Path("Florian/surrogate_models/joblibs/surrogate_model_2d_sampling_1d_training_ratio_opex_lp_upper.joblib"),
    "opex_lp_approx": Path("Florian/surrogate_models/joblibs/surrogate_model_2d_sampling_1d_training_ratio_opex_lp_approx.joblib"),
}

PRICE_2D_MODEL_PATHS = {
    "opex_milp": Path("Florian/surrogate_models/joblibs/surrogate_model_2d_sampling_1d_training_2d_opex_milp.joblib"),
    "opex_lp_lower": Path("Florian/surrogate_models/joblibs/surrogate_model_2d_sampling_1d_training_2d_opex_lp_lower.joblib"),
    "opex_lp_upper": Path("Florian/surrogate_models/joblibs/surrogate_model_2d_sampling_1d_training_2d_opex_lp_upper.joblib"),
    "opex_lp_approx": Path("Florian/surrogate_models/joblibs/surrogate_model_2d_sampling_1d_training_2d_opex_lp_approx.joblib"),
}

MODEL_FILES = {
    "MILP_pred": RESULTS_DIR / "2d_validation_no_offset_opex_milp.csv",
    "LP_Upper": RESULTS_DIR / "2d_validation_no_offset_opex_lp_upper.csv",
    "LP_Lower": RESULTS_DIR / "2d_validation_no_offset_opex_lp_lower.csv",
    "LP_Approx": RESULTS_DIR / "2d_validation_no_offset_opex_lp_approx.csv",
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
    "opex_lp_approx": "opex_lp_approx",
}


def lighter_color(color, amount=0.55):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def load_validation_predictions():
    return {model: pd.read_csv(path) for model, path in MODEL_FILES.items()}


def run_original_comparison():
    validation_data = load_validation_predictions()

    df_discrepancy_comparison = pd.concat(
        [
            validation_data["MILP_pred"]["y_test"],
            validation_data["MILP_pred"]["y_pred"],
            validation_data["LP_Upper"]["y_test"],
            validation_data["LP_Upper"]["y_pred"],
            validation_data["LP_Lower"]["y_test"],
            validation_data["LP_Lower"]["y_pred"],
            validation_data["LP_Approx"]["y_test"],
            validation_data["LP_Approx"]["y_pred"],
        ],
        axis=1,
        keys=[
            "_milp_actual",
            "_milp_pred",
            "_lp_upper_test",
            "_lp_upper_pred",
            "_lp_lower_test",
            "_lp_lower_pred",
            "_lp_approx_test",
            "_lp_approx_pred",
        ],
    )
    print(df_discrepancy_comparison.head())

    r2_milp = r2_score(
        df_discrepancy_comparison["_milp_actual"],
        df_discrepancy_comparison["_milp_pred"],
    )
    r2_upper = r2_score(
        df_discrepancy_comparison["_lp_upper_test"],
        df_discrepancy_comparison["_lp_upper_pred"],
    )
    r2_lower = r2_score(
        df_discrepancy_comparison["_lp_lower_test"],
        df_discrepancy_comparison["_lp_lower_pred"],
    )
    r2_approx = r2_score(
        df_discrepancy_comparison["_lp_approx_test"],
        df_discrepancy_comparison["_lp_approx_pred"],
    )

    r2_df = pd.DataFrame(
        [
            {
                "model": "MILP",
                "r2_actual_vs_predicted": r2_milp,
               
            },
            {
                "model": "LP_Upper",
                "r2_actual_vs_predicted": r2_upper,
                
            },
            {
                "model": "LP_Lower",
                "r2_actual_vs_predicted": r2_lower,
               
            },
            {
                "model": "LP_Approx",
                "r2_actual_vs_predicted": r2_approx,
                
            },
        ]
    )
    print(r2_df)
    r2_df.to_csv(RESULTS_DIR / "r2_score_2d_compared.csv", index=False)


def run_milp_opex_actual_comparison():
    df_actual_milp = pd.read_csv(MARIUS_RESULTS_DIR / "opex_random_sample_10.csv")
    validation_data = load_validation_predictions()

    y_actual = df_actual_milp["opex_milp"] * df_actual_milp["c_e"]  # Multipliziere mit tatsächlichem c_electricity, um die OPEX in € zu erhalten
    comparison_parts = [y_actual]
    comparison_keys = ["_milp_actual_opex"]
    r2_rows = []

    for model, df_validation in validation_data.items():
        if len(df_validation) != len(y_actual):
            raise ValueError(
                f"{model} hat {len(df_validation)} Zeilen, aber opex_random_sample_10.csv "
                f"hat {len(y_actual)} Zeilen. R2 kann so nicht sauber berechnet werden."
            )

        y_pred = df_validation["y_pred"]
        comparison_parts.append(y_pred)
        comparison_keys.append(f"_{model.lower()}_pred")
        r2_rows.append(
            {
                "model": model,
                "r2_compared_to_actual_milp_opex": r2_score(y_actual, y_pred),
            }
        )

    df_comparison = pd.concat(comparison_parts, axis=1, keys=comparison_keys)
    print(df_comparison.head())

    r2_df = pd.DataFrame(r2_rows)
    milp_r2 = r2_df.loc[r2_df["model"] == "MILP_pred", "r2_compared_to_actual_milp_opex"].iloc[0]
    r2_df["delta_to_milp_pred"] = r2_df["r2_compared_to_actual_milp_opex"] - milp_r2

    print(r2_df)
    r2_df.to_csv(RESULTS_DIR / "r2_score_2d_compared_to_actual_milp_opex_2d_no_offset.csv", index=False)


# def run_mse_vs_milp():
#     df_actual_milp = pd.read_csv(MARIUS_RESULTS_DIR / "opex_random_sample_10.csv")
#     validation_data = load_validation_predictions()

#     if MSE_MODEL not in validation_data:
#         raise ValueError(f"MSE_MODEL must be one of {list(validation_data)}.")

#     y_actual = df_actual_milp["opex_milp"]
#     y_pred = validation_data[MSE_MODEL]["y_pred"]
#     mse = mean_squared_error(y_actual, y_pred)

#     mse_df = pd.DataFrame([
#         {
#             "model": MSE_MODEL,
#             "mse_compared_to_actual_milp_opex": mse,
#         }
#     ])
#     print(mse_df)
#     mse_df.to_csv(RESULTS_DIR / f"mse_2d_{MSE_MODEL}_vs_actual_milp_opex.csv", index=False)

def plot_train_and_test_data_for_all_models():
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

    reset_plot_settings()
    fig, axes = plt.subplots(
        len(MODEL_PATHS),
        1,
        figsize=(get_figsize(16, "golden")[0], 14),
        sharex=True,
    )

    for ax, (model_name, model) in zip(axes, models.items()):
        model_label = MODEL_LABELS[model_name]
        target_column = TEST_TARGET_COLUMNS[model_name]

        x_min = min(df_train["ratio"].min(), df_test["ratio"].min())
        x_max = max(df_train["ratio"].max(), df_test["ratio"].max())
        x_line_values = np.linspace(x_min, x_max, 250)
        x_line = pd.DataFrame({"ratio": x_line_values})

        y_line = model.predict(x_line)
        intercept = model.intercept_
        slope = model.coef_[0]

        base_color = COLORS[model_name]
        train_color = lighter_color(base_color)
        test_color = base_color

        ax.plot(
            x_line_values,
            y_line,
            color=base_color,
            linewidth=2,
            label=f"{model_label}: y = {slope:.2f}x + {intercept:.2f}",
        )

        ax.scatter(
            df_train["ratio"],
            df_train[model_name],
            color=train_color,
            edgecolors=base_color,
            alpha=0.85,
            linewidths=0.7,
            marker="o",
            label="training data",
        )

        ax.scatter(
            df_test["ratio"],
            df_test[target_column],
            color=test_color,
            edgecolors="black",
            alpha=0.85,
            linewidths=0.7,
            marker="s",
            label="test data",
        )

        ax.set_ylabel("OPEX/C_el")
        ax.set_title(model_label)
        ax.grid(True)
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")

    axes[-1].set_xlabel("Ratio")
    fig.suptitle("Training and Test Data, 2D LHS sampling", y=0.995)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "train_test_regression_lines_2d_sampling_1d_training.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Training/test plot for all models saved to: {plot_path}")


def _electricity_price_column(df, preferred_columns):
    for column in preferred_columns:
        if column in df.columns:
            return column
    raise ValueError(f"None of the electricity price columns exist: {preferred_columns}")


def _concrete_opex(df, target_column, multiplier_column):
    y = df[target_column]
    if multiplier_column is None:
        return y
    return y * df[multiplier_column]


def plot_3d_price_regressions():
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    models = {name: joblib.load(path) for name, path in PRICE_2D_MODEL_PATHS.items()}

    train_electricity_column = _electricity_price_column(
        df_train,
        ["actual_c_electricity", "c_e", "c_el"],
    )
    test_electricity_column = _electricity_price_column(df_test, ["c_e", "c_el"])

    train_multiplier_column = (
        train_electricity_column if train_electricity_column == "actual_c_electricity" else None
    )
    test_multiplier_column = test_electricity_column if test_electricity_column == "c_e" else None

    c_g_min = min(df_train["c_G"].min(), df_test["c_G"].min())
    c_g_max = max(df_train["c_G"].max(), df_test["c_G"].max())
    c_el_min = min(
        df_train[train_electricity_column].min(),
        df_test[test_electricity_column].min(),
    )
    c_el_max = max(
        df_train[train_electricity_column].max(),
        df_test[test_electricity_column].max(),
    )

    c_g_grid, c_el_grid = np.meshgrid(
        np.linspace(c_g_min, c_g_max, 35),
        np.linspace(c_el_min, c_el_max, 35),
    )
    surface_input = pd.DataFrame(
        {
            "c_G": c_g_grid.ravel(),
            "c_el": c_el_grid.ravel(),
        }
    )

    reset_plot_settings()
    fig = plt.figure(figsize=(get_figsize(25, "golden")[0], 16))

    for index, (model_name, model) in enumerate(models.items(), start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        model_label = MODEL_LABELS[model_name]

        y_surface = model.predict(surface_input).reshape(c_g_grid.shape)
        y_train = _concrete_opex(df_train, model_name, train_multiplier_column)
        y_test = _concrete_opex(df_test, TEST_TARGET_COLUMNS[model_name], test_multiplier_column)

        base_color = COLORS[model_name]
        train_color = lighter_color(base_color)

        ax.plot_surface(
            c_g_grid,
            c_el_grid,
            y_surface,
            color=base_color,
            alpha=0.28,
            linewidth=0,
            antialiased=True,
        )
        ax.scatter(
            df_train["c_G"],
            df_train[train_electricity_column],
            y_train,
            color=train_color,
            edgecolors=base_color,
            linewidths=0.5,
            s=26,
            alpha=0.9,
            label="training data",
        )
        ax.scatter(
            df_test["c_G"],
            df_test[test_electricity_column],
            y_test,
            color=base_color,
            edgecolors="black",
            linewidths=0.6,
            s=36,
            marker="s",
            alpha=0.95,
            label="test data",
        )

        coefficients = model.coef_
        ax.set_title(
            f"{model_label}: z = {coefficients[0]:.2f} c_G + "
            f"{coefficients[1]:.2f} c_el + {model.intercept_:.2f}"
        )
        ax.set_xlabel("Gas price c_G")
        ax.set_ylabel("Electricity price c_el")
        ax.set_zlabel("Concrete OPEX")
        ax.legend()
        ax.view_init(elev=24, azim=-135)

    fig.suptitle("3D Price Regressions with Training and Test Data, without y-offset", y=0.98)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "3d_price_regression_planes_no_offset_all_models.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"3D price regression plot saved to: {plot_path}")



def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if RUN_MODE == "inter_model_r2":
        run_original_comparison()
    elif RUN_MODE == "compared_to_milp_r2":
        run_milp_opex_actual_comparison()
    elif RUN_MODE == "mse_vs_milp":
        run_mse_vs_milp()
    elif RUN_MODE == "plot_train_and_test_data_for_all_models":
        plot_train_and_test_data_for_all_models()
    elif RUN_MODE == "plot_3d_price_regressions":
        plot_3d_price_regressions()
    else:
        raise ValueError(
            'RUN_MODE must be "inter_model_r2", "compared_to_milp_r2", "mse_vs_milp", '
            '"plot_train_and_test_data_for_all_models", or "plot_3d_price_regressions".'
        )


if __name__ == "__main__":
    main()
