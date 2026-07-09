"""
Generate held-out test sample points (2D price pairs or 1D price ratios) and
solve the 4 canonical optimization formulations in Erdem/src/optimization/core.py
(MILP, LP lower, LP upper, LP approximated) for each point.

Test points are drawn i.i.d. uniformly at random within the feasible price
rectangle (Erdem's create_sample("random", ...)), independent of how the training
set for that mode was sampled.

The recommended entry point is run_shared(): it generates ONE random 2D price-pair
test set, solves it once, and derives the 1D test set from it exactly (no
re-solving) — so every sampling mode (1D, 2D, 2D_noY) is evaluated on the exact
same underlying price scenarios. run() generates a single mode's test set in
isolation and is kept for standalone/backward-compatible use.

Run from the repo root:  python Marius/evaluation/evaluate_on_test_samples.py
"""

from pathlib import Path

from _evaluation_common import generate_points, generate_shared_test_points, derive_1d_from_2d, solve_all

SAMPLING_MODE = "1D"  # "1D" -> price ratio (gas/electricity), "2D" -> (gas_price, electricity_price) pair
# for 2D: c_g and c_el are given in €/Mwh in output, but optimzation function calcualtes with €/Kwh! -> for post processing multiply OPEX specific with c_el/1000 
N_TEST = 10


def default_out_csv(sampling_mode: str) -> Path:
    return Path(f"Marius/results/evaluation_10_test_samples_{sampling_mode}.csv")


def run(sampling_mode: str = SAMPLING_MODE, n_test: int = N_TEST, out_csv: Path = None):
    """Generate n_test held-out test points and solve the 4 optimization formulations for them.

    Saves the result to out_csv (default: Marius/results/evaluation_test_samples_<mode>.csv)
    and returns it as a DataFrame.
    """
    out_csv = out_csv or default_out_csv(sampling_mode)

    print(f"Generating {n_test} test points ({sampling_mode})")
    points = generate_points(sampling_mode, n_test, is_train=False)

    print("Solving optimization problems for test points")
    df = solve_all(points)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")
    return df


def run_shared(n_test: int = N_TEST, out_csv_2d: Path = None, out_csv_1d: Path = None):
    """Generate ONE shared test set for every sampling mode: n_test (gas, electricity)
    price pairs drawn i.i.d. uniformly at random, solved once, saved as the 2D test
    set, and exactly derived (no re-solving) into the 1D test set.

    Returns (df_2d, df_1d).
    """
    out_csv_2d = out_csv_2d or default_out_csv("2D")
    out_csv_1d = out_csv_1d or default_out_csv("1D")

    print(f"Generating {n_test} shared test points (uniform random)")
    points = generate_shared_test_points(n_test)

    print("Solving optimization problems for shared test points")
    df_2d = solve_all(points)
    out_csv_2d.parent.mkdir(parents=True, exist_ok=True)
    df_2d.to_csv(out_csv_2d, index=False)
    print(f"Saved 2D test data to {out_csv_2d}")

    df_1d = derive_1d_from_2d(df_2d)
    out_csv_1d.parent.mkdir(parents=True, exist_ok=True)
    df_1d.to_csv(out_csv_1d, index=False)
    print(f"Saved derived 1D test data to {out_csv_1d}")

    return df_2d, df_1d


if __name__ == "__main__":
    run_shared()
