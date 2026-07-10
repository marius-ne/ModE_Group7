from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score


VALIDATION_DIR = Path("Florian/validation")
TEST_DATA_PATH = Path("Marius/results/evaluation_10_test_samples_2D.csv")
OUTPUT_CSV = VALIDATION_DIR / "absolute_opex_r2_by_training_size.csv"

TRAINING_SAMPLE_SIZES = [2, 5, 20, 40]
FILE_PATTERN = re.compile(
    r"^(?P<training_size>\d+)_train_10_test_ratio_(?P<target>opex_(?:milp|lp_upper|lp_lower|lp_approx))\.csv$"
)

MODEL_LABELS = {
    "opex_milp": "MILP",
    "opex_lp_upper": "LP upper",
    "opex_lp_lower": "LP lower",
    "opex_lp_approx": "LP approx",
}

MODEL_ORDER = ["MILP", "LP upper", "LP lower", "LP approx"]

ELECTRICITY_PRICE_COLUMNS = [
    "electricity_price_MWh",
    "actual_c_electricity",
    "c_e",
    "c_el",
    "electricity_price",
]


def electricity_price_column(test_df: pd.DataFrame) -> str:
    for column in ELECTRICITY_PRICE_COLUMNS:
        if column in test_df.columns:
            return column

    raise ValueError(
        "Keine Strompreis-Spalte in der Testdatei gefunden. "
        f"Erwartet eine von: {ELECTRICITY_PRICE_COLUMNS}"
    )


def infer_price_multiplier(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    electricity_column: str,
) -> tuple[float, str]:
    raw_price = test_df[electricity_column]
    candidates = [
        (raw_price, "raw electricity price"),
        (raw_price / 1000.0, "electricity price divided by 1000"),
    ]

    errors = []
    for multiplier, label in candidates:
        reconstructed_actual = validation_df["y_test"] * multiplier
        mean_abs_error = (reconstructed_actual - test_df[target]).abs().mean()
        errors.append((mean_abs_error, multiplier, label))

    _, best_multiplier, best_label = min(errors, key=lambda item: item[0])
    return best_multiplier, best_label


def validation_files() -> list[Path]:
    paths = []
    for path in VALIDATION_DIR.glob("*_train_10_test_ratio_opex_*.csv"):
        if path.name.startswith("40_train_40_test"):
            continue
        if FILE_PATTERN.match(path.name):
            paths.append(path)

    return sorted(paths)


def calculate_absolute_opex_r2() -> pd.DataFrame:
    test_df = pd.read_csv(TEST_DATA_PATH)
    electricity_column = electricity_price_column(test_df)

    results = pd.DataFrame(index=MODEL_ORDER, columns=[str(size) for size in TRAINING_SAMPLE_SIZES], dtype=float)

    for validation_path in validation_files():
        match = FILE_PATTERN.match(validation_path.name)
        if match is None:
            continue

        training_size = int(match.group("training_size"))
        target = match.group("target")
        if training_size not in TRAINING_SAMPLE_SIZES:
            continue
        if target not in MODEL_LABELS:
            continue
        if target not in test_df.columns:
            raise ValueError(f"Target-Spalte '{target}' fehlt in {TEST_DATA_PATH}.")

        validation_df = pd.read_csv(validation_path)
        if len(validation_df) != len(test_df):
            raise ValueError(
                f"{validation_path} hat {len(validation_df)} Zeilen, "
                f"aber {TEST_DATA_PATH} hat {len(test_df)} Zeilen."
            )

        price_multiplier, price_mode = infer_price_multiplier(
            validation_df=validation_df,
            test_df=test_df,
            target=target,
            electricity_column=electricity_column,
        )

        y_actual_absolute = test_df[target]
        y_pred_absolute = validation_df["y_pred"] * price_multiplier
        r2_absolute = r2_score(y_actual_absolute, y_pred_absolute)

        model_label = MODEL_LABELS[target]
        results.loc[model_label, str(training_size)] = r2_absolute

        print(
            f"{validation_path.name}: {model_label}, training size {training_size}, "
            f"price mode: {price_mode}, R2 absolute OPEX = {r2_absolute:.6f}"
        )

    missing_values = results.isna()
    if missing_values.any().any():
        missing = [
            f"{model}/{sample_size}"
            for model in results.index
            for sample_size in results.columns
            if missing_values.loc[model, sample_size]
        ]
        raise FileNotFoundError(f"Nicht alle erwarteten Auswertungen gefunden: {missing}")

    results.index.name = "model"
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV)
    return results


if __name__ == "__main__":
    r2_table = calculate_absolute_opex_r2()
    print(r2_table)
    print(f"Saved absolute OPEX R2 table to: {OUTPUT_CSV}")
