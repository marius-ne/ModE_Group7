from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]

TEST_DATA_PATH = ROOT / "Marius" / "results" / "evaluation_lhs_10_test_2D.csv"
MODEL_DIR = ROOT / "Florian" / "validation" / "joblibs"
OUTPUT_DIR = ROOT / "Florian" / "validation" / "results_2d_models"

SAMPLE_SIZES = (5, 20, 40)
TARGET_COLUMNS = ("opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx")
SUMMARY_COLUMNS = {
    "opex_milp": "MILP",
    "opex_lp_lower": "LP lower",
    "opex_lp_upper": "LP upper",
    "opex_lp_approx": "LP approx",
}


def model_path(sample_size: int, target: str) -> Path:
    return MODEL_DIR / f"2D_{sample_size}_2d_discrete_absolute_{target}.joblib"


def output_path(sample_size: int, target: str, test_sample_count: int) -> Path:
    return OUTPUT_DIR / f"{sample_size}_train_{test_sample_count}_test_2d_discrete_{target}.csv"


def load_test_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found: {path}")

    df = pd.read_csv(path)
    required_columns = ("gas_price_MWh", "electricity_price_MWh", *TARGET_COLUMNS)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns {missing_columns}. "
            f"Available columns are: {list(df.columns)}"
        )
    return df


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c_G": df["gas_price_MWh"] / 1000.0,
            "c_el": df["electricity_price_MWh"] / 1000.0,
        },
        index=df.index,
    )


def predict_sample_size(sample_size: int, df_test: pd.DataFrame) -> dict[str, float]:
    x_test = build_price_features(df_test)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    r2_scores = {}
    test_sample_count = len(df_test)

    for target in TARGET_COLUMNS:
        path = model_path(sample_size, target)
        if not path.exists():
            raise FileNotFoundError(f"Model joblib not found: {path}")

        model = joblib.load(path)
        y_test = df_test[target]
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores[target] = r2

        result = pd.DataFrame(
            {
                "y_test": y_test.to_numpy(),
                "y_pred": y_pred,
                "r2": r2,
            }
        )
        result_path = output_path(sample_size, target, test_sample_count)
        result.to_csv(result_path, index=False)
        print(f"{result_path.name}: R2 = {r2:.6f}")

    return r2_scores


def write_summary(all_r2_scores: dict[int, dict[str, float]]) -> Path:
    if len(all_r2_scores) == 1:
        sample_scores = next(iter(all_r2_scores.values()))
        summary = pd.DataFrame(
            [{SUMMARY_COLUMNS[target]: sample_scores[target] for target in TARGET_COLUMNS}]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "training_size": sample_size,
                    **{
                        SUMMARY_COLUMNS[target]: sample_scores[target]
                        for target in TARGET_COLUMNS
                    },
                }
                for sample_size, sample_scores in all_r2_scores.items()
            ]
        )

    summary_path = OUTPUT_DIR / "r2_scores_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict evaluation_lhs_10_test_2D.csv with no-offset 2D price "
            "regression joblibs. Test prices are converted from EUR/MWh to EUR/kWh."
        )
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        choices=SAMPLE_SIZES,
        default=None,
        help="Predict only one sample size. By default, 5, 20 and 40 are predicted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_sizes = (args.sample_size,) if args.sample_size else SAMPLE_SIZES
    df_test = load_test_data(TEST_DATA_PATH)

    all_r2_scores = {}
    for sample_size in sample_sizes:
        print(f"Predicting no-offset 2D regressions for {sample_size} training samples")
        all_r2_scores[sample_size] = predict_sample_size(sample_size, df_test)

    summary_path = write_summary(all_r2_scores)
    print(f"Saved R2 summary to: {summary_path}")


if __name__ == "__main__":
    main()
