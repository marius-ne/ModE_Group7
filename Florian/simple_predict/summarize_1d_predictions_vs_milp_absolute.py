from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]

TEST_DATA_PATH = ROOT / "Marius" / "results" / "evaluation_lhs_10_test_2D.csv"
RESULTS_DIR = ROOT / "Florian" / "validation" / "results_1d_models"
OUTPUT_PATH = RESULTS_DIR / "r2_vs_milp_summary.csv"

TARGET_COLUMNS = ("opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx")
SUMMARY_COLUMNS = {
    "opex_milp": "MILP",
    "opex_lp_lower": "LP lower",
    "opex_lp_upper": "LP upper",
    "opex_lp_approx": "LP approx",
}

RESULT_FILE_PATTERN = re.compile(
    r"(?P<sample_size>\d+)_train_(?P<test_size>\d+)_test_ratio_(?P<target>opex_.+)\.csv$"
)


def load_test_reference(path: Path) -> tuple[pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found: {path}")

    df = pd.read_csv(path)
    required_columns = ("opex_milp", "electricity_price_MWh")
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns {missing_columns}. "
            f"Available columns are: {list(df.columns)}"
        )

    electricity_price_kwh = df["electricity_price_MWh"] / 1000.0
    return df["opex_milp"], electricity_price_kwh


def discover_prediction_files(sample_size: int | None) -> list[Path]:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    paths = []
    for path in sorted(RESULTS_DIR.glob("*_train_*_test_ratio_opex_*.csv")):
        match = RESULT_FILE_PATTERN.match(path.name)
        if not match:
            continue
        target = match.group("target")
        if target not in TARGET_COLUMNS:
            continue
        if sample_size is not None and int(match.group("sample_size")) != sample_size:
            continue
        paths.append(path)

    if not paths:
        suffix = f" for sample size {sample_size}" if sample_size is not None else ""
        raise FileNotFoundError(f"No 1D prediction CSVs found in {RESULTS_DIR}{suffix}.")
    return paths


def metadata_from_prediction_path(path: Path) -> tuple[int, int, str]:
    match = RESULT_FILE_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unexpected prediction file name: {path.name}")

    return (
        int(match.group("sample_size")),
        int(match.group("test_size")),
        match.group("target"),
    )


def calculate_r2_vs_milp(
    path: Path,
    y_milp_absolute: pd.Series,
    electricity_price_kwh: pd.Series,
) -> dict[str, object]:
    sample_size, test_size, target = metadata_from_prediction_path(path)
    prediction_df = pd.read_csv(path)

    if "y_pred" not in prediction_df.columns:
        raise ValueError(
            f"{path} is missing required column 'y_pred'. "
            f"Available columns are: {list(prediction_df.columns)}"
        )
    if len(prediction_df) != len(y_milp_absolute):
        raise ValueError(
            f"{path.name} has {len(prediction_df)} rows, but {TEST_DATA_PATH.name} "
            f"has {len(y_milp_absolute)} rows."
        )
    if test_size != len(y_milp_absolute):
        raise ValueError(
            f"{path.name} says {test_size} test samples, but {TEST_DATA_PATH.name} "
            f"contains {len(y_milp_absolute)} rows."
        )

    y_pred_absolute = prediction_df["y_pred"] * electricity_price_kwh
    r2 = r2_score(y_milp_absolute, y_pred_absolute)
    print(f"{path.name}: absolute R2 vs MILP = {r2:.6f}")

    return {
        "training_size": sample_size,
        "target": target,
        "summary_column": SUMMARY_COLUMNS[target],
        "r2": r2,
    }


def write_summary(rows: list[dict[str, object]]) -> Path:
    summary_rows = []
    for sample_size in sorted({int(row["training_size"]) for row in rows}):
        sample_rows = [row for row in rows if row["training_size"] == sample_size]
        summary_row = {"training_size": sample_size}
        for target in TARGET_COLUMNS:
            summary_column = SUMMARY_COLUMNS[target]
            match = [row for row in sample_rows if row["target"] == target]
            if match:
                summary_row[summary_column] = match[0]["r2"]
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    if len(summary) == 1:
        summary = summary.drop(columns=["training_size"])

    summary.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert 1D ratio-model predictions back to absolute OPEX with "
            "electricity_price_MWh / 1000 and compare them to actual MILP OPEX."
        )
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Summarize only one training sample size. By default, all result files are used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    y_milp_absolute, electricity_price_kwh = load_test_reference(TEST_DATA_PATH)
    prediction_paths = discover_prediction_files(args.sample_size)

    rows = [
        calculate_r2_vs_milp(path, y_milp_absolute, electricity_price_kwh)
        for path in prediction_paths
    ]
    summary_path = write_summary(rows)
    print(f"Saved 1D R2-vs-MILP summary to: {summary_path}")


if __name__ == "__main__":
    main()
