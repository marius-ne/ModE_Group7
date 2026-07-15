from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "Florian" / "validation" / "results_1d_models"
TEST_DATA_PATH = ROOT / "Marius" / "results" / "evaluation_lhs_10_test_2D.csv"
OUTPUT_PATH = RESULTS_DIR / "mse_rmse_summary.csv"

TARGET_LABELS = {
    "opex_milp": "MILP",
    "opex_lp_lower": "LP lower",
    "opex_lp_upper": "LP upper",
    "opex_lp_approx": "LP approx",
}
RESULT_FILE_PATTERN = re.compile(
    r"(?P<training_size>\d+)_train_(?P<test_size>\d+)_test_ratio_"
    r"(?P<target>opex_.+)\.csv$"
)


def squared_error_metrics(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    mse = float(((actual - predicted) ** 2).mean())
    return mse, mse**0.5


def main() -> None:
    test_data = pd.read_csv(TEST_DATA_PATH)
    required_test_columns = {"electricity_price_MWh", "opex_milp"}
    missing_test_columns = required_test_columns.difference(test_data.columns)
    if missing_test_columns:
        raise ValueError(
            f"{TEST_DATA_PATH} is missing columns: {sorted(missing_test_columns)}"
        )

    electricity_price_kwh = (
        test_data["electricity_price_MWh"] / 1000.0
    ).reset_index(drop=True)
    actual_milp = test_data["opex_milp"].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    prediction_paths = sorted(
        RESULTS_DIR.glob("*_train_*_test_ratio_opex_*.csv")
    )
    for path in prediction_paths:
        match = RESULT_FILE_PATTERN.fullmatch(path.name)
        if not match or match.group("target") not in TARGET_LABELS:
            continue

        prediction = pd.read_csv(path)
        required_prediction_columns = {"y_test", "y_pred"}
        missing_prediction_columns = required_prediction_columns.difference(
            prediction.columns
        )
        if missing_prediction_columns:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing_prediction_columns)}"
            )

        test_size = int(match.group("test_size"))
        if len(prediction) != test_size or len(prediction) != len(test_data):
            raise ValueError(
                f"{path.name} contains {len(prediction)} rows and declares {test_size} "
                f"test samples, while {TEST_DATA_PATH.name} contains {len(test_data)} rows."
            )

        target = match.group("target")
        actual_absolute = (
            prediction["y_test"].reset_index(drop=True) * electricity_price_kwh
        )
        predicted_absolute = (
            prediction["y_pred"].reset_index(drop=True) * electricity_price_kwh
        )

        mse_actual, rmse_actual = squared_error_metrics(
            actual_absolute, predicted_absolute
        )
        mse_milp, rmse_milp = squared_error_metrics(actual_milp, predicted_absolute)

        rows.append(
            {
                "training_size": int(match.group("training_size")),
                "test_size": test_size,
                "model": TARGET_LABELS[target],
                "mse_actual_vs_predicted": mse_actual,
                "rmse_actual_vs_predicted": rmse_actual,
                "mse_milp_opex_vs_predicted": mse_milp,
                "rmse_milp_opex_vs_predicted": rmse_milp,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No prediction CSVs found in {RESULTS_DIR}.")

    model_order = {label: index for index, label in enumerate(TARGET_LABELS.values())}
    summary = pd.DataFrame(rows)
    summary["_model_order"] = summary["model"].map(model_order)
    summary = summary.sort_values(["training_size", "_model_order"]).drop(
        columns="_model_order"
    )
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(summary)} metric rows to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
