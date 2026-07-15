"""Calculate the mean absolute MILP OPEX from an evaluated 2D sample CSV."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_CSV = ROOT / "Marius/results/evaluation_lhs_10_test_2D.csv"


def calculate_mean_milp_opex(sample_csv: str | Path) -> float:
    """Return the arithmetic mean of the absolute ``opex_milp`` values."""
    sample_csv = Path(sample_csv)
    data = pd.read_csv(sample_csv)

    if "opex_milp" not in data.columns:
        raise ValueError(
            f"{sample_csv} does not contain the required 'opex_milp' column."
        )

    milp_opex = pd.to_numeric(data["opex_milp"], errors="raise")
    if milp_opex.empty:
        raise ValueError(f"{sample_csv} does not contain any MILP OPEX values.")
    if milp_opex.isna().any():
        raise ValueError(f"{sample_csv} contains missing MILP OPEX values.")

    return float(milp_opex.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the mean absolute MILP OPEX of an evaluated sample."
    )
    parser.add_argument(
        "sample_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_SAMPLE_CSV,
        help=f"Evaluated 2D sample CSV (default: {DEFAULT_SAMPLE_CSV})",
    )
    args = parser.parse_args()

    mean_opex = calculate_mean_milp_opex(args.sample_csv)
    print(f"Mean absolute MILP OPEX: {mean_opex:,.2f} EUR")


if __name__ == "__main__":
    main()
