from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


RESULTS_DIR = Path("Florian/results/regression")
TRAINING_DATA_PATH = Path("Marius/results/evaluation_log_samples.csv")
TEST_DATA_PATH = Path("Marius/results/opex_random_sample_10.csv")

TARGET_COLUMNS = {
    "MILP": {
        "train": "opex_milp",
        "test": "opex_milp",
    },
    "LP_Upper": {
        "train": "opex_lp_upper",
        "test": "opex_lp_upper",
    },
    "LP_Lower": {
        "train": "opex_lp_lower",
        "test": "opex_lp_lower",
    },
    "LP_Approx": {
        "train": "opex_lp_approx",
        "test": "opex_lp_approximated",
    },
}


def fit_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def check_mse_against_milp():
    df_train = pd.read_csv(TRAINING_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    x_train = df_train[["ratio"]]
    x_test = df_test[["ratio"]]
    y_milp_train = df_train["opex_milp"]
    y_milp_test = df_test["opex_milp"]

    rows = []
    for model_name, columns in TARGET_COLUMNS.items():
        model = fit_model(x_train, df_train[columns["train"]])

        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        rows.append(
            {
                "model": model_name,
                "mse_vs_milp_on_training_data": mean_squared_error(
                    y_milp_train,
                    train_pred,
                ),
                "mse_vs_milp_on_test_data": mean_squared_error(
                    y_milp_test,
                    test_pred,
                ),
                "intercept": model.intercept_,
                "ratio_coefficient": model.coef_[0],
            }
        )

    df_result = pd.DataFrame(rows)

    print("\nMSE gegen MILP auf den Trainingsdaten:")
    print(
        df_result[["model", "mse_vs_milp_on_training_data"]]
        .sort_values("mse_vs_milp_on_training_data")
        .to_string(index=False)
    )

    print("\nMSE gegen MILP auf den Testdaten:")
    print(
        df_result[["model", "mse_vs_milp_on_test_data"]]
        .sort_values("mse_vs_milp_on_test_data")
        .to_string(index=False)
    )

    output_path = RESULTS_DIR / "mse_models_vs_milp_train_and_test.csv"
    df_result.to_csv(output_path, index=False)
    print(f"\nErgebnisse gespeichert unter: {output_path}")


if __name__ == "__main__":
    check_mse_against_milp()
