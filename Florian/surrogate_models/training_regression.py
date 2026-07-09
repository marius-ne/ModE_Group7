import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Choose which input should be used for training:
# "ratio"     -> previous behavior: one feature, ratio = c_G / c_el
# "prices_2d" -> two features, concrete gas and electricity prices: c_G and c_el
TRAINING_MODE = "ratio"  # "ratio" or "prices_2d"

RATIO_TRAINING_FILE = "Marius/results/evaluation_5_training_samples_1D.csv"
RATIO_TEST_FILE = "Marius/results/evaluation_10_test_samples_1D.csv"

PRICE_2D_TRAINING_FILE = "Marius/results/opex_discrete_prices_lhs_40.csv"

REGRESSION_RESULTS_DIR = "Florian/validation"
COMPARISON_2D_DIR = f"{REGRESSION_RESULTS_DIR}"
JOBLIB_DIR = "Florian/validation/joblibs"

OPEX_COLUMNS = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]

RATIO_TARGET_COLUMNS = {
    "opex_milp": "opex_milp",
    "opex_lp_lower": "opex_lp_lower",
    "opex_lp_upper": "opex_lp_upper",
    "opex_lp_approx": "opex_lp_approx",
}

RATIO_TEST_TARGET_COLUMNS = {
    "opex_milp": "opex_milp",
    "opex_lp_lower": "opex_lp_lower",
    "opex_lp_upper": "opex_lp_upper",
    "opex_lp_approx": "opex_lp_approx",
}

PRICE_2D_TARGET_COLUMNS = {
    "opex_milp": "opex_milp",
    "opex_lp_lower": "opex_lp_lower",
    "opex_lp_upper": "opex_lp_upper",
    "opex_lp_approx": "opex_lp_approx",
}


def ensure_output_dirs():
    os.makedirs(REGRESSION_RESULTS_DIR, exist_ok=True)
    os.makedirs(COMPARISON_2D_DIR, exist_ok=True)
    os.makedirs(JOBLIB_DIR, exist_ok=True)


def train_and_save_regressions(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        train_feature_columns: list[str],
        test_feature_columns: list[str],
        train_target_columns: dict[str, str],
        test_target_columns: dict[str, str],
        output_dir: str,
        output_suffix: str,
        test_target_multiplier_column: str | None = None,
        fit_intercept: bool = True,
):
    x_train = df_train[train_feature_columns]
    x_test = df_test[test_feature_columns].copy()
    x_test.columns = train_feature_columns

    for target in OPEX_COLUMNS:
        print(f"\n{'=' * 50}")
        print(f"Starte Modelltraining fuer: {target}")
        print(f"{'=' * 50}")

        y_train = df_train[train_target_columns[target]]
        #if train_target_multiplier_column is not None:
        #    y_train = y_train #* df_train[train_target_multiplier_column]

        y_test = df_test[test_target_columns[target]]
        if test_target_multiplier_column is not None:
            y_test = y_test * df_test[test_target_multiplier_column]
        
        regression_model = LinearRegression(fit_intercept=fit_intercept)
        regression_model.fit(x_train, y_train)

        y_pred = regression_model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse:,.2f}")
        print(f"R^2 Score: {r2:.4f}")
        print(f"Basiswert (Intercept): {regression_model.intercept_:,.2f}")
        for feature_name, coefficient in zip(train_feature_columns, regression_model.coef_):
            print(f"Koeffizient {feature_name}: {coefficient:,.2f}")

        validation_df = pd.DataFrame({
            "y_test": y_test.to_numpy(),
            "y_pred": y_pred,
            "r2": r2,
        })
        validation_path = f"{output_dir}/5_train_10_test_{output_suffix}_{target}.csv"
        validation_df.to_csv(validation_path, index=False)
        print(f"Validation gespeichert unter: {validation_path}")

        joblib_path = f"{JOBLIB_DIR}/5_{output_suffix}_{target}.joblib"
        joblib.dump(regression_model, joblib_path)
        print(f"Modell gespeichert unter: {joblib_path}")


def run_ratio_training():
    df_train = pd.read_csv(RATIO_TRAINING_FILE)
    df_test = pd.read_csv(RATIO_TEST_FILE)

    print(f"Verwende Ratio-Training: {RATIO_TRAINING_FILE}")
    train_and_save_regressions(
        df_train=df_train,
        df_test=df_test,
        train_feature_columns=["ratio"],
        test_feature_columns=["ratio"],
        train_target_columns=RATIO_TARGET_COLUMNS,
        test_target_columns=RATIO_TEST_TARGET_COLUMNS,
        output_dir=REGRESSION_RESULTS_DIR,
        output_suffix="ratio",
    )


def run_2d_price_training():
    df_train = pd.read_csv(PRICE_2D_TRAINING_FILE)
    df_test = pd.read_csv(RATIO_TEST_FILE)

    print(f"Verwende 2D-Preis-Training: {PRICE_2D_TRAINING_FILE}")
    train_and_save_regressions(
        df_train=df_train,
        df_test=df_test,
        train_feature_columns=["c_G", "c_el"],
        test_feature_columns=["c_G", "c_e"],
        train_target_columns=PRICE_2D_TARGET_COLUMNS,
        test_target_columns=PRICE_2D_TARGET_COLUMNS,
        output_dir=COMPARISON_2D_DIR,
        output_suffix="2d",
        test_target_multiplier_column="c_e",
        fit_intercept=False,
    )


def main():
    ensure_output_dirs()

    if TRAINING_MODE == "ratio":
        run_ratio_training()
    elif TRAINING_MODE == "prices_2d":
        run_2d_price_training()
    else:
        raise ValueError('TRAINING_MODE must be either "ratio" or "prices_2d".')


if __name__ == "__main__":
    main()
