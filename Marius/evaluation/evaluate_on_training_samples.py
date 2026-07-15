"""
Generate training sample points (2D price pairs or 1D price ratios) and solve
the 4 canonical optimization formulations in Erdem/src/optimization/core.py
(MILP, LP lower, LP upper, LP approximated) for each point.

Run from the repo root:  python Marius/evaluation/evaluate_on_training_samples.py
"""

from pathlib import Path

from _evaluation_common import generate_points, solve_all

SAMPLING_MODE = "2D"  # "1D" -> price ratio (gas/electricity), "2D" -> (gas_price, electricity_price) pair
N_TRAIN = 5
SAMPLING_METHOD_2D = "lhs"  # training sampling method for 2D mode ("sobol", "lhs" or "random")
SAMPLING_METHOD_1D = "angle"  # training sampling method for 1D mode ("log" or "angle")


def default_out_csv(sampling_mode: str) -> Path:
    return Path(f"Marius/results/2D_evaluation_{N_TRAIN}_training_samples_{sampling_mode}.csv")


def run(sampling_mode: str = SAMPLING_MODE, n_train: int = N_TRAIN, method_2d: str = SAMPLING_METHOD_2D,
        method_1d: str = SAMPLING_METHOD_1D, out_csv: Path = None):
    """Generate n_train training points and solve the 4 optimization formulations for them.

    Saves the result to out_csv (default: Marius/results/evaluation_{N_TRAIN}_training_samples_{sampling_mode}.csv)
    and returns it as a DataFrame.
    """
    out_csv = out_csv or default_out_csv(sampling_mode)

    print(f"Generating {n_train} training points ({sampling_mode})")
    points = generate_points(sampling_mode, n_train, is_train=True, method_2d=method_2d, method_1d=method_1d)

    print("Solving optimization problems for training points")
    df = solve_all(points)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")
    return df


if __name__ == "__main__":
    run()
