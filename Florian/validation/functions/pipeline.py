from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR_1D = REPO_ROOT / "Florian" / "validation" / "joblibs"
TEST_SAMPLE_1D = REPO_ROOT / "Marius" / "results" / "evaluation_lhs_10_test_1D.csv"
OUTPUT_CSV = REPO_ROOT / "Florian" / "validation" / "results_1d_models" / "r2_specific_1d_vs_actual_test_sample.csv"
MODEL_PATTERN = "*_ratio_opex_*.joblib"


def find_1d_models() -> list[Path]:
    if not MODEL_DIR_1D.exists():
        raise FileNotFoundError(f"1D model directory not found: {MODEL_DIR_1D}")

    paths = sorted(MODEL_DIR_1D.glob(MODEL_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No 1D ratio models found in: {MODEL_DIR_1D}")
    return paths


def load_1d_test_sample() -> pd.DataFrame:
    if not TEST_SAMPLE_1D.exists():
        raise FileNotFoundError(f"1D test sample not found: {TEST_SAMPLE_1D}")

    df = pd.read_csv(TEST_SAMPLE_1D)
    expected = ["ratio", "opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"1D test sample is missing columns: {missing}")
    return df.reset_index(drop=True)


def parse_model_file(path: Path) -> tuple[int, str]:
    stem = path.stem
    if not stem.endswith("_ratio_opex") and "_ratio_opex_" not in stem:
        raise ValueError(f"Unexpected 1D model file name: {path.name}")

    prefix, target = stem.split("_ratio_opex_", maxsplit=1)
    try:
        training_size = int(prefix)
    except ValueError as exc:
        raise ValueError(f"Unexpected training-size prefix in 1D model file: {path.name}") from exc
    return training_size, target


def score_model(model_path: Path, test_df: pd.DataFrame) -> dict[str, object]:
    training_size, target = parse_model_file(model_path)
    model = joblib.load(model_path)
    X = test_df[["ratio"]].values

    y_actual = test_df["opex_milp"].values
    y_pred = model.predict(X)
    return {
        "training_size": training_size,
        "target": target,
        "model_file": model_path.name,
        "r2_predicted_vs_actual_milp": float(r2_score(y_actual, y_pred)),
        "n_samples": len(y_actual),
    }


def save_summary(rows: list[dict[str, object]]) -> Path:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["training_size", "target"]).reset_index(drop=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    return OUTPUT_CSV


def main() -> None:
    test_df = load_1d_test_sample()
    model_paths = find_1d_models()

    print(f"Using 1D test sample: {TEST_SAMPLE_1D}")
    print(f"Loaded {len(test_df)} test samples and {len(model_paths)} models.")

    rows = [score_model(path, test_df) for path in model_paths]
    output_path = save_summary(rows)

    print(f"Saved 1D specific OPEX R² summary to: {output_path}")


if __name__ == "__main__":
    main()
