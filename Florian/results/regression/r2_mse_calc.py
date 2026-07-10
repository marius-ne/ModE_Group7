from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


# Auswahlmoeglichkeit:
# "inter_model_r2" = bisherige Auswertung wie zuvor
# "compared_to_milp_r2" = alle y_pred gegen das echte MILP-OPEX aus MILP_TEST_FILE
# "mse_vs_milp" = MILP-Testwerte mit y_pred eines waehlbaren Modells per MSE vergleichen
RUN_MODE = "compared_to_milp_r2"  # "inter_model_r2", "compared_to_milp_r2", "mse_vs_milp"
# MSE_MODEL = "MILP_pred"
# USE_ACTUAL_MILP_OPEX = True

RESULTS_DIR = Path("Florian/validation")
MARIUS_RESULTS_DIR = Path("Marius/results")
MILP_TEST_FILE = MARIUS_RESULTS_DIR / "evaluation_10_test_samples_1D.csv"


def test_sample_size(test_file: Path = MILP_TEST_FILE) -> int:
    return len(pd.read_csv(test_file))


def model_files(test_size = "5"):
 

    return {
        "MILP_pred": RESULTS_DIR / f"{test_size}_train_10_test_ratio_opex_milp.csv",
        "LP_Upper": RESULTS_DIR / f"{test_size}_train_10_test_ratio_opex_lp_upper.csv",
        "LP_Lower": RESULTS_DIR / f"{test_size}_train_10_test_ratio_opex_lp_lower.csv",
        "LP_Approx": RESULTS_DIR / f"{test_size}_train_10_test_ratio_opex_lp_approx.csv",
    }


def load_validation_predictions():
    return {model: pd.read_csv(path) for model, path in model_files().items()}


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
    r2_df.to_csv(RESULTS_DIR / f"{test_sample_size()}_r2_score_compared.csv", index=False)


def run_milp_opex_actual_comparison():
    df_actual_milp = pd.read_csv(MILP_TEST_FILE)
    validation_data = load_validation_predictions()

    y_actual = df_actual_milp["opex_milp"]
    comparison_parts = [y_actual]
    comparison_keys = ["_milp_actual_opex"]
    r2_rows = []

    for model, df_validation in validation_data.items():
        if len(df_validation) != len(y_actual):
            raise ValueError(
                f"{model} hat {len(df_validation)} Zeilen, aber {MILP_TEST_FILE} "
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
    r2_df.to_csv(
        RESULTS_DIR / f"{len(df_actual_milp)}_r2_score_compared_to_milp.csv",
        index=False,
    )


def compare_milp_test_values_to_model_predictions(model_name, use_actual_milp_opex=True):
    files = model_files()
    if model_name not in files:
        raise ValueError(f"model_name muss eines von {list(files)} sein.")

    df_validation = pd.read_csv(files[model_name])

    if use_actual_milp_opex:
        df_milp = pd.read_csv(MILP_TEST_FILE)
        y_milp_test = df_milp["opex_milp"]
    else:
        y_milp_test = pd.read_csv(files["MILP_pred"])["y_test"]

    if len(df_validation) != len(y_milp_test):
        raise ValueError(
            f"{model_name} hat {len(df_validation)} Zeilen, aber die MILP-Testwerte "
            f"haben {len(y_milp_test)} Zeilen. MSE kann so nicht sauber berechnet werden."
        )

    y_pred = df_validation["y_pred"]
    mse = mean_squared_error(y_milp_test, y_pred)

    print(f"MSE MILP-Testwerte vs. {model_name}-Predictions: {mse:,.2f}")
    return mse


if RUN_MODE == "inter_model_r2":
    run_original_comparison()
elif RUN_MODE == "compared_to_milp_r2":
    run_milp_opex_actual_comparison()
elif RUN_MODE == "mse_vs_milp":
    compare_milp_test_values_to_model_predictions(
        MSE_MODEL,
        use_actual_milp_opex=USE_ACTUAL_MILP_OPEX,
    )
else:
    raise ValueError(
        'RUN_MODE muss entweder "inter_model_r2", "compared_to_milp_r2" '
        'oder "mse_vs_milp" sein.'
    )
