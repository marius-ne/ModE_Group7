"""
Evaluate an existing test-sample CSV with the same solve_all() logic used by
evaluate_on_test_samples.py.

Run from the repo root:
    python Marius/evaluation/evaluate_test_sample_file.py
"""

import argparse
from pathlib import Path

import pandas as pd

from _evaluation_common import solve_all

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Input configuration
# ============================================================
# Set this to the CSV file you want to evaluate.
SAMPLE_CSV = ROOT / "Erdem/results/Sampling/test/lhs_10_samples.csv"

# Optional: set a custom output file. If None, the result is written to
# Marius/results/evaluation_<input_file_name>.csv.
OUT_CSV = None

# Optional: set to "1D" or "2D" if the mode should not be inferred from columns.
# In "1D", gas/electricity prices from the sample are converted from EUR/MWh
# to EUR/kWh first, then ratio = c_gas / c_el is evaluated with c_el = 1.
SAMPLING_MODE = "1D"


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def load_test_sample_points(sample_csv: str | Path, sampling_mode: str | None = None) -> pd.DataFrame:
    """Load a test sample CSV and normalize it for solve_all().

    Accepted price columns are gas_price/electricity_price or
    gas_price_MWh/electricity_price_MWh. Accepted direct 1D columns are ratio or
    ratios. If sampling_mode="1D" and price columns exist, ratios are calculated
    from prices after converting EUR/MWh to EUR/kWh. solve_all() then evaluates
    those ratios with c_el=1 and c_gas=ratio.
    """
    sample_csv = repo_path(sample_csv)
    samples = pd.read_csv(sample_csv)

    gas_col = _first_existing_column(samples, ["gas_price_MWh", "gas_price"])
    electricity_col = _first_existing_column(samples, ["electricity_price_MWh", "electricity_price"])
    ratio_col = _first_existing_column(samples, ["ratio", "ratios"])

    if sampling_mode is None:
        if gas_col and electricity_col:
            sampling_mode = "2D"
        elif ratio_col:
            sampling_mode = "1D"
        else:
            raise ValueError(
                f"Cannot infer sample type from {sample_csv}. "
                f"Need 2D columns gas_price/electricity_price or 1D column ratio/ratios. "
                f"Available columns: {list(samples.columns)}"
            )

    if sampling_mode == "2D":
        if not gas_col or not electricity_col:
            raise ValueError(
                f"2D sample {sample_csv} needs gas_price/electricity_price "
                f"or gas_price_MWh/electricity_price_MWh. Available columns: {list(samples.columns)}"
            )
        return pd.DataFrame({
            "gas_price_MWh": samples[gas_col].astype(float),
            "electricity_price_MWh": samples[electricity_col].astype(float),
        })

    if sampling_mode == "1D":
        if gas_col and electricity_col:
            gas_price_kwh = samples[gas_col].astype(float) / 1000.0
            electricity_price_kwh = samples[electricity_col].astype(float) / 1000.0
            if (electricity_price_kwh == 0).any():
                raise ValueError(f"Cannot calculate ratio from {sample_csv}: electricity price contains zero.")
            return pd.DataFrame({"ratio": gas_price_kwh / electricity_price_kwh})

        if not ratio_col:
            raise ValueError(
                f"1D sample {sample_csv} needs gas/electricity price columns or ratio/ratios. "
                f"Available columns: {list(samples.columns)}"
            )
        return pd.DataFrame({"ratio": samples[ratio_col].astype(float)})

    raise ValueError(f"Unknown sampling_mode '{sampling_mode}', expected '1D', '2D' or None.")


def default_out_csv(sample_csv: str | Path) -> Path:
    sample_csv = Path(sample_csv)
    return ROOT / "Marius/results" / f"evaluation_{sample_csv.stem}_1D.csv"


def evaluate_test_sample_file(
        sample_csv: str | Path,
        out_csv: str | Path = None,
        sampling_mode: str | None = None,
) -> pd.DataFrame:
    """Evaluate an existing test-sample CSV and save the solved OPEX table."""
    sample_csv = repo_path(sample_csv)
    out_csv = repo_path(out_csv) if out_csv is not None else default_out_csv(sample_csv)

    points = load_test_sample_points(sample_csv, sampling_mode=sampling_mode)
    detected_mode = "2D" if "gas_price_MWh" in points.columns else "1D"
    print(f"Loaded {len(points)} {detected_mode} test points from {sample_csv}")

    print("Solving optimization problems for provided test points")
    df = solve_all(points)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an existing test-sample CSV with MILP/LP formulations."
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        help="Existing test-sample CSV to evaluate. Overrides SAMPLE_CSV at the top of this file.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Where to save evaluated results. Defaults to Marius/results/evaluation_<sample_stem>.csv.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["1D", "2D"],
        help="Optional override. If omitted, the mode is inferred from the sample columns.",
    )
    args = parser.parse_args()

    sample_csv = args.sample_csv or SAMPLE_CSV
    out_csv = args.out_csv if args.out_csv is not None else OUT_CSV
    sampling_mode = args.sampling_mode if args.sampling_mode is not None else SAMPLING_MODE

    evaluate_test_sample_file(
        sample_csv=sample_csv,
        out_csv=out_csv,
        sampling_mode=sampling_mode,
    )


if __name__ == "__main__":
    main()
