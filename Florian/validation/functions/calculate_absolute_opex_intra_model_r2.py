from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[3]
PREDICTION_DIR = REPO_ROOT / "Florian" / "validation" / "results_1d_models"
TEST_SAMPLE_PATH = REPO_ROOT / "Marius" / "results" / "evaluation_lhs_10_test_2D.csv"
OUTPUT_DIR = REPO_ROOT / "Florian" / "validation" / "absolute_opex_intra_model_r2"
OUTPUT_FILE_NAME = "absolute_opex_intra_model_r2_summary.csv"

PREDICTION_FILE_PATTERN = re.compile(
    r"^(?P<training_size>\d+)_train_10_test_ratio_"
    r"(?P<target>opex_(?:milp|lp_upper|lp_lower|lp_approx))\.csv$"
)

MODEL_LABELS = {
    "opex_milp": "MILP",
    "opex_lp_upper": "LP upper",
    "opex_lp_lower": "LP lower",
    "opex_lp_approx": "LP approx",
}


def find_prediction_files() -> list[Path]:
    if not PREDICTION_DIR.exists():
        raise FileNotFoundError(f"Prediction directory not found: {PREDICTION_DIR}")

    prediction_files = [
        path
        for path in PREDICTION_DIR.glob("*_train_10_test_ratio_opex_*.csv")
        if PREDICTION_FILE_PATTERN.match(path.name)
    ]
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in: {PREDICTION_DIR}")

    return sorted(prediction_files)


def load_electricity_prices_kwh() -> pd.Series:
    if not TEST_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Test sample file not found: {TEST_SAMPLE_PATH}")

    test_sample = pd.read_csv(TEST_SAMPLE_PATH)
    electricity_column = "electricity_price_MWh"
    if electricity_column not in test_sample.columns:
        raise ValueError(f"Column '{electricity_column}' is missing in {TEST_SAMPLE_PATH}")

    return test_sample[electricity_column] / 1000.0


def calculate_file_r2(prediction_path: Path, electricity_prices_kwh: pd.Series) -> dict[str, object]:
    match = PREDICTION_FILE_PATTERN.match(prediction_path.name)
    if match is None:
        raise ValueError(f"Unexpected prediction file name: {prediction_path.name}")

    prediction_df = pd.read_csv(prediction_path)
    expected_columns = {"y_test", "y_pred"}
    missing_columns = expected_columns.difference(prediction_df.columns)
    if missing_columns:
        raise ValueError(f"{prediction_path} is missing columns: {sorted(missing_columns)}")

    if len(prediction_df) != len(electricity_prices_kwh):
        raise ValueError(
            f"{prediction_path} has {len(prediction_df)} rows, but "
            f"{TEST_SAMPLE_PATH} has {len(electricity_prices_kwh)} rows."
        )

    y_test_absolute_opex = prediction_df["y_test"] * electricity_prices_kwh
    y_pred_absolute_opex = prediction_df["y_pred"] * electricity_prices_kwh

    target = match.group("target")
    return {
        "training_size": int(match.group("training_size")),
        "target": target,
        "model": MODEL_LABELS[target],
        "prediction_file": prediction_path.name,
        "r2_absolute_opex_intra_model": float(r2_score(y_test_absolute_opex, y_pred_absolute_opex)),
        "n_samples": len(prediction_df),
    }


def calculate_absolute_opex_intra_model_r2() -> pd.DataFrame:
    electricity_prices_kwh = load_electricity_prices_kwh()
    rows = [
        calculate_file_r2(prediction_path, electricity_prices_kwh)
        for prediction_path in find_prediction_files()
    ]

    summary = pd.DataFrame(rows)
    return summary.sort_values(["training_size", "target"]).reset_index(drop=True)


def create_output_dir() -> Path:
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        return OUTPUT_DIR

    suffix = 2
    while True:
        candidate = OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_{suffix}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        suffix += 1


def main() -> None:
    summary = calculate_absolute_opex_intra_model_r2()
    output_dir = create_output_dir()
    output_csv = output_dir / OUTPUT_FILE_NAME
    summary.to_csv(output_csv, index=False)

    print(summary)
    print(f"Saved absolute OPEX intra-model R2 summary to: {output_csv}")


if __name__ == "__main__":
    main()
