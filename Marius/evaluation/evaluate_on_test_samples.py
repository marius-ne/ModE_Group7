"""
Generate ONE shared 2D price-pair test set (drawn from the feasible price rectangle)
and solve the 4 canonical optimization formulations in Erdem/src/optimization/core.py
for it, then derive the corresponding 1D (ratio -> specific OPEX) test set from it via
the exact specific_opex = absolute_opex / c_el identity (see
_evaluation_common.derive_1d_from_2d) -- no re-solving, so every mode (1D or 2D) is
evaluated on the exact same underlying price scenarios.

Run from the repo root:  python Marius/evaluation/evaluate_on_test_samples.py
"""

from pathlib import Path

from _evaluation_common import generate_shared_test_points, solve_all, derive_1d_from_2d, TEST_SEED

N_TEST = 10
TEST_METHOD_2D = "lhs"  # "lhs", "sobol" or "random"
N_CORNER = 0
N_EDGES = 0


def run_shared(n: int, out_csv_2d: Path, out_csv_1d: Path, test_method_2d: str = TEST_METHOD_2D,
               n_corner: int = N_CORNER, n_edges: int = N_EDGES):
    """Generate n shared 2D test price pairs, solve the 4 formulations for them, save to
    out_csv_2d, then derive and save the corresponding 1D (ratio -> specific OPEX) test
    set to out_csv_1d (no re-solving).

    Returns (df_2d, df_1d).
    """
    print(f"Generating {n} shared 2D test points ({test_method_2d}, corners={n_corner}, edges={n_edges})")
    points = generate_shared_test_points(n, method_2d=test_method_2d, n_corner=n_corner, n_edges=n_edges)

    print("Solving optimization problems for test points")
    df_2d = solve_all(points)

    out_csv_2d.parent.mkdir(parents=True, exist_ok=True)
    df_2d.to_csv(out_csv_2d, index=False)
    print(f"Saved 2D test set to {out_csv_2d}")

    df_1d = derive_1d_from_2d(df_2d)
    out_csv_1d.parent.mkdir(parents=True, exist_ok=True)
    df_1d.to_csv(out_csv_1d, index=False)
    print(f"Saved derived 1D test set to {out_csv_1d}")

    return df_2d, df_1d


if __name__ == "__main__":
    # Same filename convention as run_full_pipeline.test_tag(), so a set generated here is
    # picked up (and not silently mismatched) by the pipeline.
    tag = f"{N_TEST}_{TEST_METHOD_2D}_interior_only_seed{TEST_SEED}"
    run_shared(
        N_TEST,
        out_csv_2d=Path(f"Marius/results/evaluation_test_samples_{tag}_2D.csv"),
        out_csv_1d=Path(f"Marius/results/evaluation_test_samples_{tag}_1D.csv"),
    )
