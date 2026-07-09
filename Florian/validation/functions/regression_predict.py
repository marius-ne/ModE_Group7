from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import r2_score


# ============================================================
# User settings
# ============================================================
# Choose one:
# "ratios"          -> use one feature: ratio = gas price / electricity price
# "discrete_prices" -> use two features: gas price and electricity price
PREDICTION_MODE = "ratios"

# The script also accepts command line arguments, but these values are the
# easiest place to change paths when running the file directly from an IDE.
RATIO_MODEL_DIR = "Florian/validation/joblibs"
DISCRETE_PRICE_MODEL_DIR = "Florian/surrogate_models/joblibs"
RATIO_MODEL_PREFIX = "2"

RATIO_TEST_SAMPLE_CSV = "Marius/results/evaluation_10_test_samples_1D.csv"
DISCRETE_PRICE_TEST_SAMPLE_CSV = "Marius/results/evaluation_10_test_samples_2D.csv"

OUTPUT_CSV = None


OPEX_TARGETS = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]
VALID_MODES = ["ratios", "discrete_prices"]

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModeConfig:
    model_dir: Path
    test_sample_csv: Path
    model_patterns: tuple[str, ...]
    output_suffix: str


def as_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_mode_configs() -> dict[str, ModeConfig]:
    return {
        "ratios": ModeConfig(
            model_dir=as_repo_path(RATIO_MODEL_DIR),
            test_sample_csv=as_repo_path(RATIO_TEST_SAMPLE_CSV),
            model_patterns=(
                f"{RATIO_MODEL_PREFIX}_ratio_{{target}}.joblib",
            ),
            output_suffix="ratio",
        ),
        "discrete_prices": ModeConfig(
            model_dir=as_repo_path(DISCRETE_PRICE_MODEL_DIR),
            test_sample_csv=as_repo_path(DISCRETE_PRICE_TEST_SAMPLE_CSV),
            model_patterns=(
                "surrogate_model_2d_prices_40_{target}.joblib",
                "surrogate_model_no_offset_2d_{target}.joblib",
                "surrogate_model_2d_sampling_1d_training_2d_{target}.joblib",
                "_2d_{target}.joblib",
            ),
            output_suffix="2d",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict OPEX values for a test sample with saved regression models "
            "and export actual values, predicted values and R2 scores to CSV."
        )
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=PREDICTION_MODE,
        help='Prediction mode. Default is the PREDICTION_MODE setting at the top of the file.',
    )
    parser.add_argument(
        "--ratio-model-dir",
        default=RATIO_MODEL_DIR,
        help="Directory containing ratio regression .joblib files.",
    )
    parser.add_argument(
        "--discrete-price-model-dir",
        default=DISCRETE_PRICE_MODEL_DIR,
        help="Directory containing two-price regression .joblib files.",
    )
    parser.add_argument(
        "--test-sample",
        default=None,
        help="CSV with test samples. If omitted, the default CSV for the selected mode is used.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_CSV,
        help="Optional combined output CSV path. If omitted, only the per-model CSVs and R2 summary are written.",
    )
    return parser.parse_args()


def update_config_from_args(args: argparse.Namespace) -> ModeConfig:
    configs = build_mode_configs()
    selected = configs[args.mode]

    model_dir = (
        as_repo_path(args.ratio_model_dir)
        if args.mode == "ratios"
        else as_repo_path(args.discrete_price_model_dir)
    )
    test_sample_csv = as_repo_path(args.test_sample) if args.test_sample else selected.test_sample_csv

    return ModeConfig(
        model_dir=model_dir,
        test_sample_csv=test_sample_csv,
        model_patterns=selected.model_patterns,
        output_suffix=selected.output_suffix,
    )


def find_model_file(model_dir: Path, patterns: tuple[str, ...], target: str) -> Path:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    for pattern in patterns:
        candidate = model_dir / pattern.format(target=target)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No .joblib model found for target '{target}' in {model_dir}. "
        f"Expected one of: {[pattern.format(target=target) for pattern in patterns]}"
    )


def source_column_for_feature(df: pd.DataFrame, feature: str) -> str:
    aliases = {
        "ratio": ["ratio"],
        "c_G": ["c_G", "gas_price", "gas_price_MWh"],
        # c_e and actual_c_electricity are preferred over c_el because several
        # test files keep c_el fixed at 1 while c_e stores the real price.
        "c_el": ["actual_c_electricity", "c_e", "c_el", "electricity_price", "electricity_price_MWh"],
    }

    for column in aliases.get(feature, [feature]):
        if column in df.columns:
            return column

    raise ValueError(
        f"Missing feature '{feature}' in test sample. Available columns are: {list(df.columns)}"
    )


def feature_values(df: pd.DataFrame, feature: str, source_column: str) -> pd.Series:
    values = df[source_column]
    if feature in {"c_G", "c_el"} and source_column.endswith("_MWh"):
        return values / 1000.0
    return values


def build_model_input(df: pd.DataFrame, model) -> pd.DataFrame:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        n_features = getattr(model, "n_features_in_", None)
        if n_features == 1:
            feature_names = ["ratio"]
        elif n_features == 2:
            feature_names = ["c_G", "c_el"]
        else:
            raise ValueError(
                "The model does not store feature_names_in_ and its number of features "
                f"({n_features}) is not supported."
            )

    model_input = pd.DataFrame(index=df.index)
    feature_mapping = {}
    for feature in list(feature_names):
        source_column = source_column_for_feature(df, feature)
        feature_mapping[feature] = source_column
        model_input[feature] = feature_values(df, feature, source_column)

    print(f"Feature mapping: {feature_mapping}")
    return model_input


def predict_target(df_test: pd.DataFrame, target: str, model_path: Path) -> pd.DataFrame:
    if target not in df_test.columns:
        raise ValueError(
            f"Missing actual target column '{target}' in test sample. "
            f"Available columns are: {list(df_test.columns)}"
        )
    if "opex_milp" not in df_test.columns:
        raise ValueError(
            "Missing MILP reference column 'opex_milp' in test sample. "
            f"Available columns are: {list(df_test.columns)}"
        )

    model = joblib.load(model_path)
    x_test = build_model_input(df_test, model)
    y_actual = df_test[target]
    y_milp = df_test["opex_milp"]
    y_predicted = model.predict(x_test)
    r2 = r2_score(y_actual, y_predicted)
    r2_vs_milp = r2_score(y_milp, y_predicted)

    sample_columns = df_test.drop(columns=OPEX_TARGETS, errors="ignore").reset_index(drop=True)
    result = pd.DataFrame(
        {
            "sample_index": df_test.index,
            "target": target,
            "actual": y_actual.to_numpy(),
            "predicted": y_predicted,
            "r2_actual_vs_predicted": r2,
            "r2_milp_actual_vs_predicted": r2_vs_milp,
            "model_file": str(model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path),
        }
    )
    result = pd.concat([sample_columns, result], axis=1)

    print(
        f"{target}: R2 actual vs predicted = {r2:.6f}, "
        f"R2 MILP actual vs predicted = {r2_vs_milp:.6f} using {model_path.name}"
    )
    return result


def write_model_validation_csv(
    result: pd.DataFrame,
    target: str,
    output_dir: Path,
    sample_size: int,
    output_suffix: str,
) -> Path:
    validation_df = pd.DataFrame(
        {
            "y_test": result["actual"].to_numpy(),
            "y_pred": result["predicted"].to_numpy(),
            "r2": result["r2_actual_vs_predicted"].iloc[0],
        }
    )
    output_path = output_dir / f"{sample_size}_2_{output_suffix}_{target}.csv"
    validation_df.to_csv(output_path, index=False)
    print(f"Saved model validation to: {output_path}")
    return output_path


def write_r2_summary(prediction_frames: list[pd.DataFrame], output_dir: Path, sample_size: int) -> Path:
    model_names = {
        "opex_milp": "MILP_pred",
        "opex_lp_upper": "LP_Upper",
        "opex_lp_lower": "LP_Lower",
        "opex_lp_approx": "LP_Approx",
    }
    rows = []
    for result in prediction_frames:
        target = result["target"].iloc[0]
        rows.append(
            {
                "model": model_names[target],
                "r2_compared_to_actual_milp_opex": result["r2_milp_actual_vs_predicted"].iloc[0],
            }
        )

    r2_df = pd.DataFrame(rows)
    milp_r2 = r2_df.loc[r2_df["model"] == "MILP_pred", "r2_compared_to_actual_milp_opex"].iloc[0]
    r2_df["delta_to_milp_pred"] = r2_df["r2_compared_to_actual_milp_opex"] - milp_r2

    output_path = output_dir / f"{sample_size}_r2_score_compared_to_milp.csv"
    r2_df.to_csv(output_path, index=False)
    print(f"Saved R2 summary to: {output_path}")
    return output_path


def run_prediction(mode: str, config: ModeConfig, output_csv: Path | None) -> list[Path]:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use one of: {VALID_MODES}")
    if not config.test_sample_csv.exists():
        raise FileNotFoundError(f"Test sample CSV does not exist: {config.test_sample_csv}")

    print(f"Mode: {mode}")
    print(f"Model directory: {config.model_dir}")
    print(f"Test sample: {config.test_sample_csv}")
    if output_csv is not None:
        print(f"Combined output CSV: {output_csv}")

    df_test = pd.read_csv(config.test_sample_csv)
    prediction_frames = []
    output_dir = ROOT / "Florian" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_size = len(df_test)
    written_paths = []

    for target in OPEX_TARGETS:
        model_path = find_model_file(config.model_dir, config.model_patterns, target)
        result = predict_target(df_test, target, model_path)
        prediction_frames.append(result)
        written_paths.append(
            write_model_validation_csv(
                result=result,
                target=target,
                output_dir=output_dir,
                sample_size=sample_size,
                output_suffix=config.output_suffix,
            )
        )

    written_paths.append(write_r2_summary(prediction_frames, output_dir, sample_size))

    if output_csv is not None:
        output = pd.concat(prediction_frames, ignore_index=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_csv, index=False)
        print(f"Saved combined predictions to: {output_csv}")
        written_paths.append(output_csv)

    return written_paths


def main() -> None:
    args = parse_args()
    config = update_config_from_args(args)
    output_csv = as_repo_path(args.output) if args.output else None
    run_prediction(args.mode, config, output_csv)


if __name__ == "__main__":
    main()
