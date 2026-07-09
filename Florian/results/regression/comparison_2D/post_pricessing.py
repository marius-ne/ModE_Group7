from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score


RESULTS_DIR = Path("Florian/results/regression/comparison_2D")
TEST_DATA_PATH = Path("Marius/results/opex_random_sample_10.csv")

VALIDATION_PREFIX = "2d_sampling_1d_training_validation_ratio_specific_"
VALIDATION_PATTERN = f"{VALIDATION_PREFIX}opex_*.csv"

# The ratio-specific predictions are converted back to absolute OPEX with c_e.
ELECTRICITY_PRICE_COLUMN = "c_e"

OUTPUT_SUMMARY_FILE = RESULTS_DIR / "r2_absolute_opex_2d_sampling_1d_training_ratio_specific.csv"


def target_from_validation_file(path: Path) -> str:
    return path.stem.removeprefix(VALIDATION_PREFIX)


def validate_same_length(validation_df: pd.DataFrame, test_df: pd.DataFrame, path: Path) -> None:
    if len(validation_df) != len(test_df):
        raise ValueError(
            f"{path.name} hat {len(validation_df)} Zeilen, "
            f"aber {TEST_DATA_PATH} hat {len(test_df)} Zeilen."
        )


def calculate_absolute_opex_r2() -> pd.DataFrame:
    test_df = pd.read_csv(TEST_DATA_PATH)

    if ELECTRICITY_PRICE_COLUMN not in test_df.columns:
        raise ValueError(
            f"Spalte '{ELECTRICITY_PRICE_COLUMN}' fehlt in {TEST_DATA_PATH}."
        )

    validation_paths = sorted(RESULTS_DIR.glob(VALIDATION_PATTERN))
    if not validation_paths:
        raise FileNotFoundError(
            f"Keine Dateien gefunden mit Pattern: {RESULTS_DIR / VALIDATION_PATTERN}"
        )

    r2_rows = []

    for validation_path in validation_paths:
        target = target_from_validation_file(validation_path)
        if target not in test_df.columns:
            raise ValueError(
                f"Target-Spalte '{target}' aus {validation_path.name} fehlt in {TEST_DATA_PATH}."
            )

        validation_df = pd.read_csv(validation_path)
        validate_same_length(validation_df, test_df, validation_path)

        y_actual_absolute = test_df[target] * test_df[ELECTRICITY_PRICE_COLUMN]
        y_pred_absolute = validation_df["y_pred"] * test_df[ELECTRICITY_PRICE_COLUMN]

        r2_absolute = r2_score(y_actual_absolute, y_pred_absolute)

        r2_rows.append(
            {
                "target": target,
                "validation_file": validation_path.name,
                "r2_absolute_opex": r2_absolute,
            }
        )

    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)
    return r2_df


def main():
    r2_df = calculate_absolute_opex_r2()
    print(r2_df)
    print(f"R2 summary saved to: {OUTPUT_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
