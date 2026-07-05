from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


# Auswahlmoeglichkeit:
# "inter_model_r2" = bisherige Auswertung wie zuvor
# "compared_to_milp_r2" = alle y_pred gegen das echte MILP-OPEX aus Marius/results/opex_random_sample_10.csv
# "mse_vs_milp" = MILP-Testwerte mit y_pred eines waehlbaren Modells per MSE vergleichen
RUN_MODE = "compared_to_milp_r2"  # "inter_model_r2", "compared_to_milp_r2", "mse_vs_milp"
MSE_MODEL = "MILP_pred"

RESULTS_DIR = Path("Florian/results/regression/comparison_2D")
MARIUS_RESULTS_DIR = Path("Marius/results")

MODEL_FILES = {
    "MILP_pred": RESULTS_DIR / "validation_1d_trainingratio_opex_milp.csv",
    "LP_Upper": RESULTS_DIR / "validation_1d_trainingratio_opex_lp_upper.csv",
    "LP_Lower": RESULTS_DIR / "validation_1d_trainingratio_opex_lp_lower.csv",
    "LP_Approx": RESULTS_DIR / "validation_1d_trainingratio_opex_lp_approx.csv",
}


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

    y_actual = df_actual_milp["opex_milp"]#* df_actual_milp["c_e"]  # Multipliziere mit tatsächlichem c_electricity, um die OPEX in € zu erhalten
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
    r2_df.to_csv(RESULTS_DIR / "r2_score_2d_compared_to_actual_milp_opex_1d_training.csv", index=False)


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if RUN_MODE == "inter_model_r2":
        run_original_comparison()
    elif RUN_MODE == "compared_to_milp_r2":
        run_milp_opex_actual_comparison()
    elif RUN_MODE == "mse_vs_milp":
        run_mse_vs_milp()
    else:
        raise ValueError(
            'RUN_MODE must be "inter_model_r2", "compared_to_milp_r2", or "mse_vs_milp".'
        )


if __name__ == "__main__":
    main()
