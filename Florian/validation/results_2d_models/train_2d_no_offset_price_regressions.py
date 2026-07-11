from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[2]

TRAINING_DIR = ROOT / "Marius" / "results"
OUTPUT_DIR = ROOT / "Florian" / "validation" / "joblibs"

SAMPLE_SIZES = (5, 20, 40)
TARGET_COLUMNS = ("opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx")
FEATURE_COLUMNS = ("gas_price_MWh", "electricity_price_MWh")
KWH_FEATURE_COLUMNS = ("c_G", "c_el")


def training_csv_path(sample_size: int) -> Path:
    return TRAINING_DIR / f"2D_evaluation_{sample_size}_training_samples_2D.csv"


def output_joblib_path(sample_size: int, target: str) -> Path:
    return OUTPUT_DIR / f"2D_{sample_size}_2d_discrete_absolute_{target}.joblib"


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [
        column
        for column in (*FEATURE_COLUMNS, *TARGET_COLUMNS)
        if column not in df.columns
    ]
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


def train_no_offset_model(x_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression(fit_intercept=False)
    model.fit(x_train, y_train)
    return model


def train_sample_size(sample_size: int) -> list[Path]:
    training_path = training_csv_path(sample_size)
    df = load_training_data(training_path)
    x_train = build_price_features(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written_paths = []

    for target in TARGET_COLUMNS:
        model = train_no_offset_model(x_train, df[target])
        output_path = output_joblib_path(sample_size, target)
        joblib.dump(model, output_path)
        written_paths.append(output_path)
        print(
            f"{output_path.name}: {target} = "
            f"{model.coef_[0]:.6f} * c_G + {model.coef_[1]:.6f} * c_el"
        )

    return written_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train 2D linear OPEX regressions without y-offset from gas and "
            "electricity prices converted to EUR/kWh."
        )
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        choices=SAMPLE_SIZES,
        default=None,
        help="Train only one sample size. By default, 5, 20 and 40 are trained.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_sizes = (args.sample_size,) if args.sample_size else SAMPLE_SIZES

    for sample_size in sample_sizes:
        print(f"Training no-offset 2D regressions for {sample_size} samples")
        train_sample_size(sample_size)


if __name__ == "__main__":
    main()
