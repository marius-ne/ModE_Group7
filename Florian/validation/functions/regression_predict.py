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
# "pipeline"       -> predict with every .joblib model in the selected joblibs directory
# "ratios"          -> use one feature: ratio = gas price / electricity price
# "discrete_prices" -> use two features: gas price and electricity price
PREDICTION_MODE = "pipeline"

# The script also accepts command line arguments, but these values are the
# easiest place to change paths when running the file directly from an IDE.
PIPELINE_MODEL_DIR = "Florian"
PIPELINE_OUTPUT_DIR = "Florian/validation/pipeline_predictions"
RATIO_MODEL_DIR = "Florian/validation/joblibs"
DISCRETE_PRICE_MODEL_DIR = "Florian/validation/joblibs"
RATIO_MODEL_PREFIX = "2"

RATIO_TEST_SAMPLE_CSV = "Marius/results/evaluation_lhs_10_test_1D.csv"
DISCRETE_PRICE_TEST_SAMPLE_CSV = "Marius/results/evaluation_lhs_10_test_2D.csv"

OUTPUT_CSV = None


OPEX_TARGETS = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]
VALID_MODES = ["pipeline", "ratios", "discrete_prices"]

ROOT = Path(__file__).resolve().parents[3]


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
                "2D_40_2d_discrete_absolute_{target}.joblib",
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
        "--pipeline-model-dir",
        default=PIPELINE_MODEL_DIR,
        help=(
            "Directory containing .joblib files for pipeline mode. "
            "If this points to Florian, all joblibs directories below it are scanned."
        ),
    )
    parser.add_argument(
        "--pipeline-output-dir",
        default=PIPELINE_OUTPUT_DIR,
        help="Directory where pipeline mode writes per-model predictions and the R2 summary.",
    )
    parser.add_argument(
        "--pipeline-ratio-test-sample",
        default=RATIO_TEST_SAMPLE_CSV,
        help="CSV used by pipeline mode for one-feature ratio models.",
    )
    parser.add_argument(
        "--pipeline-discrete-price-test-sample",
        default=DISCRETE_PRICE_TEST_SAMPLE_CSV,
        help="CSV used by pipeline mode for two-feature price models.",
    )
    parser.add_argument(
        "--test-sample",
        default=None,
        help=(
            "CSV with test samples. If omitted, the default CSV for the selected mode is used. "
            "In pipeline mode this overrides both pipeline test sample paths."
        ),
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


def discover_model_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    if model_dir.name.lower() == "florian":
        model_paths = sorted(model_dir.glob("**/joblibs/*.joblib"))
    else:
        model_paths = sorted(model_dir.glob("*.joblib"))

    if not model_paths:
        raise FileNotFoundError(f"No .joblib models found in: {model_dir}")

    return model_paths


def infer_target_from_model_path(model_path: Path) -> str:
    matches = [target for target in OPEX_TARGETS if target in model_path.stem]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Cannot infer OPEX target from model file name: {model_path.name}. "
            f"Expected one of {OPEX_TARGETS} in the file name."
        )
    raise ValueError(f"Model file name matches multiple targets: {model_path.name} -> {matches}")


def feature_names_for_model(model) -> list[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    n_features = getattr(model, "n_features_in_", None)
    if n_features == 1:
        return ["ratio"]
    if n_features == 2:
        return ["c_G", "c_el"]

    raise ValueError(
        "The model does not store feature_names_in_ and its number of features "
        f"({n_features}) is not supported."
    )


def prediction_mode_for_features(feature_names: list[str]) -> str:
    feature_set = set(feature_names)
    if feature_set == {"ratio"}:
        return "ratios"
    if feature_set in (
        {"c_G", "c_el"},
        {"gas_price_MWh", "electricity_price_MWh"},
        {"gas_price", "electricity_price"},
    ):
        return "discrete_prices"

    raise ValueError(
        f"Cannot choose a test sample for model features {feature_names}. "
        'Supported feature sets are ["ratio"], ["c_G", "c_el"] and price-column pairs.'
    )


def source_column_for_feature(df: pd.DataFrame, feature: str) -> str:
    aliases = {
        "ratio": ["ratio"],
        "c_G": ["c_G", "gas_price", "gas_price_MWh"],
        # c_e and actual_c_electricity are preferred over c_el because several
        # test files keep c_el fixed at 1 while c_e stores the real price.
        "c_el": ["actual_c_electricity", "c_e", "c_el", "electricity_price", "electricity_price_MWh"],
        "gas_price_MWh": ["gas_price_MWh", "c_G", "gas_price"],
        "electricity_price_MWh": ["electricity_price_MWh", "c_el", "c_e", "actual_c_electricity", "electricity_price"],
        "gas_price": ["gas_price", "gas_price_MWh", "c_G"],
        "electricity_price": ["electricity_price", "electricity_price_MWh", "c_el", "c_e", "actual_c_electricity"],
    }

    for column in aliases.get(feature, [feature]):
        if column in df.columns:
            return column

    raise ValueError(
        f"Missing feature '{feature}' in test sample. Available columns are: {list(df.columns)}"
    )


def model_expects_kwh_prices(model) -> bool:
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        return False

    return max(abs(float(coefficient)) for coefficient in coefficients) > 10_000


def feature_values(df: pd.DataFrame, feature: str, source_column: str, model) -> pd.Series:
    values = df[source_column]
    if source_column.endswith("_MWh") and model_expects_kwh_prices(model):
        return values / 1000.0
    return values


def build_model_input(df: pd.DataFrame, model) -> pd.DataFrame:
    feature_names = feature_names_for_model(model)

    model_input = pd.DataFrame(index=df.index)
    feature_mapping = {}
    for feature in feature_names:
        source_column = source_column_for_feature(df, feature)
        feature_mapping[feature] = source_column
        model_input[feature] = feature_values(df, feature, source_column, model)

    print(f"Feature mapping: {feature_mapping}")
    return model_input


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
            "model_file": display_path(model_path),
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


def predict_model_file(
    model_path: Path,
    test_samples_by_mode: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, object]]:
    model = joblib.load(model_path)
    feature_names = feature_names_for_model(model)
    mode = prediction_mode_for_features(feature_names)
    test_sample_csv = test_samples_by_mode[mode]
    target = infer_target_from_model_path(model_path)

    if not test_sample_csv.exists():
        raise FileNotFoundError(f"Test sample CSV does not exist: {test_sample_csv}")

    df_test = pd.read_csv(test_sample_csv)
    result = predict_loaded_model(
        df_test=df_test,
        target=target,
        model=model,
        model_path=model_path,
        prediction_mode=mode,
        test_sample_csv=test_sample_csv,
    )
    summary = {
        "model_file": display_path(model_path),
        "target": target,
        "prediction_mode": mode,
        "features": ",".join(feature_names),
        "test_sample": display_path(test_sample_csv),
        "n_test_samples": len(df_test),
        "r2_actual_vs_predicted": result["r2_actual_vs_predicted"].iloc[0],
        "r2_milp_actual_vs_predicted": result["r2_milp_actual_vs_predicted"].iloc[0],
    }
    return result, summary


def predict_loaded_model(
    df_test: pd.DataFrame,
    target: str,
    model,
    model_path: Path,
    prediction_mode: str,
    test_sample_csv: Path,
) -> pd.DataFrame:
    if target not in df_test.columns:
        raise ValueError(
            f"Missing actual target column '{target}' in test sample {test_sample_csv}. "
            f"Available columns are: {list(df_test.columns)}"
        )
    if "opex_milp" not in df_test.columns:
        raise ValueError(
            f"Missing MILP reference column 'opex_milp' in test sample {test_sample_csv}. "
            f"Available columns are: {list(df_test.columns)}"
        )

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
            "prediction_mode": prediction_mode,
            "test_sample": display_path(test_sample_csv),
            "model_file": display_path(model_path),
        }
    )
    result = pd.concat([sample_columns, result], axis=1)

    print(
        f"{model_path.name}: {target}, mode={prediction_mode}, "
        f"R2 actual vs predicted = {r2:.6f}, "
        f"R2 MILP actual vs predicted = {r2_vs_milp:.6f}"
    )
    return result


def write_pipeline_outputs(
    prediction_frames: list[pd.DataFrame],
    summary_rows: list[dict[str, object]],
    output_dir: Path,
    output_csv: Path | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []

    for result in prediction_frames:
        model_stem = Path(result["model_file"].iloc[0]).stem
        output_path = output_dir / f"{model_stem}_test_predictions.csv"
        result.to_csv(output_path, index=False)
        written_paths.append(output_path)
        print(f"Saved pipeline predictions to: {output_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "pipeline_r2_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    written_paths.append(summary_path)
    print(f"Saved pipeline R2 summary to: {summary_path}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(prediction_frames, ignore_index=True).to_csv(output_csv, index=False)
        written_paths.append(output_csv)
        print(f"Saved combined pipeline predictions to: {output_csv}")

    return written_paths


def run_pipeline_prediction(
    model_dir: Path,
    ratio_test_sample_csv: Path,
    discrete_price_test_sample_csv: Path,
    output_dir: Path,
    output_csv: Path | None,
) -> list[Path]:
    model_paths = discover_model_files(model_dir)
    test_samples_by_mode = {
        "ratios": ratio_test_sample_csv,
        "discrete_prices": discrete_price_test_sample_csv,
    }

    print("Mode: pipeline")
    print(f"Model directory: {model_dir}")
    print(f"Found {len(model_paths)} model file(s).")
    print(f"Ratio test sample: {ratio_test_sample_csv}")
    print(f"Discrete-price test sample: {discrete_price_test_sample_csv}")

    prediction_frames = []
    summary_rows = []
    for model_path in model_paths:
        result, summary = predict_model_file(model_path, test_samples_by_mode)
        prediction_frames.append(result)
        summary_rows.append(summary)

    return write_pipeline_outputs(
        prediction_frames=prediction_frames,
        summary_rows=summary_rows,
        output_dir=output_dir,
        output_csv=output_csv,
    )


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
    output_csv = as_repo_path(args.output) if args.output else None

    if args.mode == "pipeline":
        override_test_sample = as_repo_path(args.test_sample) if args.test_sample else None
        ratio_test_sample_csv = override_test_sample or as_repo_path(args.pipeline_ratio_test_sample)
        discrete_price_test_sample_csv = override_test_sample or as_repo_path(
            args.pipeline_discrete_price_test_sample
        )
        run_pipeline_prediction(
            model_dir=as_repo_path(args.pipeline_model_dir),
            ratio_test_sample_csv=ratio_test_sample_csv,
            discrete_price_test_sample_csv=discrete_price_test_sample_csv,
            output_dir=as_repo_path(args.pipeline_output_dir),
            output_csv=output_csv,
        )
        return

    config = update_config_from_args(args)
    run_prediction(args.mode, config, output_csv)


if __name__ == "__main__":
    main()
