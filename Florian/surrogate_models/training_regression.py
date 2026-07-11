import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Choose which input should be used for training:
# "ratio"     -> previous behavior: one feature, ratio = c_G / c_el
# "prices_2d" -> two features, concrete gas and electricity prices: c_G and c_el
TRAINING_MODE = "prices_2d"  # "ratio" or "prices_2d"

RATIO_TRAINING_FILE = "Marius/results/evaluation_training_samples_1D_angle.csv"
RATIO_TEST_FILE = "Marius/results/evaluation_lhs_10_test_1D.csv"

PRICE_2D_TRAINING_FILE = "Marius/results/2D_evaluation_5_training_samples_2D.csv"
PRICE_2D_TEST_FILE = "Marius/results/2D_evaluation_10_test_2D.csv"

REGRESSION_RESULTS_DIR = "Florian/validation/results_2d_models"
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

FEATURE_ALIASES = {
    "ratio": ["ratio"],
    "c_G": ["c_G", "gas_price_MWh", "gas_price"],
    "c_el": ["c_el", "c_e", "electricity_price_MWh", "electricity_price", "actual_c_electricity"],
}


def ensure_output_dirs():
    os.makedirs(REGRESSION_RESULTS_DIR, exist_ok=True)
    os.makedirs(COMPARISON_2D_DIR, exist_ok=True)
    os.makedirs(JOBLIB_DIR, exist_ok=True)


def find_feature_column(df: pd.DataFrame, canonical_feature: str) -> str:
    for column in FEATURE_ALIASES.get(canonical_feature, [canonical_feature]):
        if column in df.columns:
            return column

    raise ValueError(
        f"Keine Spalte fuer Feature '{canonical_feature}' gefunden. "
        f"Erwartet eine von {FEATURE_ALIASES.get(canonical_feature, [canonical_feature])}. "
        f"Vorhandene Spalten: {list(df.columns)}"
    )


def build_feature_frame(df: pd.DataFrame, canonical_features: list[str]) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    column_mapping = {}

    for feature in canonical_features:
        source_column = find_feature_column(df, feature)
        column_mapping[feature] = source_column
        features[feature] = df[source_column]

    print(f"Feature-Spaltenzuordnung: {column_mapping}")
    return features


def validate_3d_regression_without_offset(
        feature_columns: list[str],
        fit_intercept: bool,
) -> None:
    if set(feature_columns) == {"c_G", "c_el"} and fit_intercept:
        raise ValueError(
            "2D-Preis-Training erzeugt eine Regressionsebene im 3D-Raum. "
            "Ein Offset/Intercept ist hier verboten: fit_intercept muss False sein."
        )


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
    validate_3d_regression_without_offset(train_feature_columns, fit_intercept)
    x_train = build_feature_frame(df_train, train_feature_columns)
    x_test = build_feature_frame(df_test, test_feature_columns)
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
        if set(train_feature_columns) == {"c_G", "c_el"} and abs(regression_model.intercept_) > 1e-9:
            raise RuntimeError(
                f"Offset fuer 3D-Regression ist nicht null ({regression_model.intercept_})."
            )

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
        validation_path = f"{output_dir}/2D_40_train_10_test_{output_suffix}_{target}.csv"
        validation_df.to_csv(validation_path, index=False)
        print(f"Validation gespeichert unter: {validation_path}")

        joblib_path = f"{JOBLIB_DIR}/2D_40_{output_suffix}_{target}.joblib"
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


def run_2d_price_training_with_discrete_absolute_test():
    df_train = pd.read_csv(PRICE_2D_TRAINING_FILE)
    df_test = pd.read_csv(PRICE_2D_TEST_FILE)

    print(f"Verwende 2D-Preis-Training: {PRICE_2D_TRAINING_FILE}")
    print(f"Teste gegen diskrete Preise mit absoluten OPEX-Werten: {PRICE_2D_TEST_FILE}")
    train_and_save_regressions(
        df_train=df_train,
        df_test=df_test,
        train_feature_columns=["c_G", "c_el"],
        test_feature_columns=["c_G", "c_el"],
        train_target_columns=PRICE_2D_TARGET_COLUMNS,
        test_target_columns=PRICE_2D_TARGET_COLUMNS,
        output_dir=COMPARISON_2D_DIR,
        output_suffix="2d_discrete_absolute",
        fit_intercept=False,
    )


def main():
    ensure_output_dirs()

    if TRAINING_MODE == "ratio":
        run_ratio_training()
    elif TRAINING_MODE == "prices_2d":
        run_2d_price_training_with_discrete_absolute_test()
    else:
        raise ValueError('TRAINING_MODE must be either "ratio" or "prices_2d".')


if __name__ == "__main__":
    main()
