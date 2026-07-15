"""
Shared helpers for evaluate_on_training_samples.py and evaluate_on_test_samples.py:
generating price sample points (2D price pairs or 1D price ratios) and solving
the 4 canonical optimization formulations in Erdem/src/optimization/core.py for
each point.
"""

import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))
from src.sampling.core import create_sample
from src.optimization.core import solve_milp, solve_lp_lower, solve_lp_upper, solve_lp_approximated

MIP_GAP = 1e-2
STRICT_DEMAND_SATISFACTION = True
LP_APPROX_MODE = "mean_efficiency"
C_EL_REF = 1.0  # reference electricity price [€/kWh] used to derive c_gas = ratio * C_EL_REF in 1D mode

# Seed for the test sample's LHS/Sobol sampler. Deliberately different from create_sample's
# default (28), which the 2D *training* sample uses -- otherwise test and training points
# come out of the same quasi-random sequence and are not an independent draw.
TEST_SEED = 4711

OPEX_COLUMNS = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]

# Wall-clock seconds each formulation took on each sample point, recorded by solve_all
# alongside the OPEX it produced. The solves happen anyway, so timing them here is free and
# means no downstream script (e.g. Marius/visualization/plot_pareto_accuracy_vs_time.py)
# ever has to re-solve just to find out how long a formulation takes.
TIME_COLUMNS = ["time_milp", "time_lp_lower", "time_lp_upper", "time_lp_approx"]

# Number of branch-and-bound LP relaxations Gurobi explored to solve the MILP (see
# solve_milp's docstring in Erdem/src/optimization/core.py), recorded alongside the OPEX
# it produced -- a property of that one MILP solve, not of the units its OPEX is in.
MILP_NUM_LPS_COLUMN = "milp_num_lps"

_demand_df = pd.read_csv(ROOT / "energy_demands.csv")
Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()


def generate_points(
        sampling_mode: str,
        n: int,
        is_train: bool,
        method_2d: str = "sobol",
        method_1d: str = "log",
        test_method_2d: str = "lhs",
        test_n_corner: int = 0,
        test_n_edges: int = 0,
) -> pd.DataFrame:
    """Generate n sample points as either (gas_price_MWh, electricity_price_MWh) pairs or price ratios.

    :param sampling_mode: "2D" for price pairs, "1D" for price ratios.
    :param is_train: training points use Sobol/LHS/log/angle; test points use
        test_method_2d within the price rectangle (see generate_shared_test_points).
    :param method_2d: sampling method for 2D training mode ("sobol", "lhs" or "random").
    :param method_1d: sampling method for 1D training mode ("log" or "angle").
    :param test_method_2d: sampling method for 2D test mode ("sobol", "lhs" or "random").
    :param test_n_corner: number of corner points to force into 2D test samples.
    :param test_n_edges: number of edge-midpoint points to force into 2D test samples.
    """
    if sampling_mode == "2D":
        if is_train:
            df, _ = create_sample(method_2d, n)
        else:
            return generate_shared_test_points(
                n,
                method_2d=test_method_2d,
                n_corner=test_n_corner,
                n_edges=test_n_edges,
            )
        points = df[["gas_price", "electricity_price"]].reset_index(drop=True)
        return points.rename(columns={"gas_price": "gas_price_MWh", "electricity_price": "electricity_price_MWh"})
    elif sampling_mode == "1D":
        if is_train:
            ratios = create_sample(method_1d, n)
        else:
            df = generate_shared_test_points(
                n,
                method_2d=test_method_2d,
                n_corner=test_n_corner,
                n_edges=test_n_edges,
            )
            ratios = df["gas_price_MWh"] / df["electricity_price_MWh"]
        return pd.DataFrame({"ratio": ratios.reset_index(drop=True)})
    else:
        raise ValueError(f"Unknown sampling_mode '{sampling_mode}', expected '1D' or '2D'.")


def generate_shared_test_points(
        n: int,
        method_2d: str = "lhs",
        n_corner: int = 0,
        n_edges: int = 0,
        seed: int = TEST_SEED,
) -> pd.DataFrame:
    """Generate n (gas_price_MWh, electricity_price_MWh) pairs with the chosen
    2D test sampling method, seeded with TEST_SEED so the points are an independent
    draw from the training sample's (which uses create_sample's default seed).

    This is the single test-point generator every sampling mode's test set is built
    from (see derive_1d_from_2d), so 1D, 2D and 2D_noY are all evaluated on the exact
    same underlying price scenarios.
    """
    df, _ = create_sample(method_2d, n, n_corner=n_corner, n_edges=n_edges, seed=seed)
    points = df[["gas_price", "electricity_price"]].reset_index(drop=True)
    return points.rename(columns={"gas_price": "gas_price_MWh", "electricity_price": "electricity_price_MWh"})


def derive_1d_from_2d(df_2d: pd.DataFrame) -> pd.DataFrame:
    """Derive the 1D (ratio -> specific OPEX) test set from an already-solved 2D
    (price pair -> absolute OPEX) test set, using the exact identity
    specific_opex = absolute_opex / c_el (see solve_all's docstring for why this holds
    — OPEX is homogeneous of degree 1 in prices). No re-solving needed, and the
    underlying price scenarios are identical to the 2D test set.
    """
    c_el = df_2d["electricity_price_MWh"] / 1000.0
    df_1d = pd.DataFrame({"ratio": df_2d["gas_price_MWh"] / df_2d["electricity_price_MWh"]})
    for col in OPEX_COLUMNS:
        df_1d[col] = df_2d[col] / c_el
    # Solve times and the MILP node count carry over verbatim: they are properties of the
    # 2D solve these rows were derived from, not values that scale with an electricity price.
    for col in [*TIME_COLUMNS, MILP_NUM_LPS_COLUMN]:
        if col in df_2d.columns:
            df_1d[col] = df_2d[col]
    return df_1d


def _timed(solve, *args, **kwargs) -> tuple[float, float]:
    """Run a solver and return (opex, wall-clock seconds it took)."""
    start = perf_counter()
    opex = solve(*args, **kwargs)[0]
    return opex, perf_counter() - start


def solve_all(points: pd.DataFrame) -> pd.DataFrame:
    """Solve all 4 optimization problems for every sample point, return points + opex columns
    + the wall-clock seconds each solve took (TIME_COLUMNS).

    In 2D mode, c_g/c_el are the real sampled prices, so the returned opex_* columns are
    absolute OPEX in €. In 1D mode, c_el is pinned to the arbitrary reference C_EL_REF
    (not a real price), so the returned opex_* columns are actually *specific* OPEX
    (OPEX per unit electricity price, i.e. OPEX / c_el, units €/(€/kWh)) — to get the
    absolute OPEX for a real c_el, multiply the returned value by that real c_el.

    The time_* columns are plain wall-clock seconds and are NOT rescaled by c_el anywhere:
    how long a solve takes is a property of the solver, not of the units its answer is
    reported in. Likewise milp_num_lps (the number of branch-and-bound LP relaxations
    Gurobi explored to solve the MILP) is a property of that solve, not of c_el.
    """
    is_2d = "gas_price_MWh" in points.columns
    opex_kind = "absolute OPEX [€]" if is_2d else "specific OPEX [€/(€/kWh)]"
    rows = []
    for i, row in points.iterrows():
        if is_2d:
            c_g = row["gas_price_MWh"] / 1000.0
            c_el = row["electricity_price_MWh"] / 1000.0
            ratio = c_g / c_el
        else:
            c_el = C_EL_REF
            c_g = row["ratio"] * c_el
            ratio = row["ratio"]

        print(f"  [{i + 1}/{len(points)}] c_g={c_g:.5f} €/kWh  c_el={c_el:.5f} €/kWh  r={ratio:.5f}  ({opex_kind})")
        milp_start = perf_counter()
        opex_milp, dispatch_milp = solve_milp(
            Q_D, P_D, c_g, c_el,
            mip_gap=MIP_GAP, strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )
        t_milp = perf_counter() - milp_start
        milp_num_lps = dispatch_milp.attrs.get("milp_num_lps")
        opex_lower, t_lower = _timed(
            solve_lp_lower, Q_D, P_D, c_g, c_el,
            strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )
        opex_upper, t_upper = _timed(
            solve_lp_upper, Q_D, P_D, c_g, c_el,
            strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )
        opex_approx, t_approx = _timed(
            solve_lp_approximated, Q_D, P_D, c_g, c_el,
            mode=LP_APPROX_MODE, strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )

        rows.append({
            "opex_milp": opex_milp,
            "opex_lp_lower": opex_lower,
            "opex_lp_upper": opex_upper,
            "opex_lp_approx": opex_approx,
            "time_milp": t_milp,
            "time_lp_lower": t_lower,
            "time_lp_upper": t_upper,
            "time_lp_approx": t_approx,
            MILP_NUM_LPS_COLUMN: milp_num_lps,
        })
        print(f"    MILP={opex_milp:,.2f}  LP_lower={opex_lower:,.2f}  "
              f"LP_upper={opex_upper:,.2f}  LP_approx={opex_approx:,.2f}")
        print(f"    times [s]: MILP={t_milp:.3f}  LP_lower={t_lower:.3f}  "
              f"LP_upper={t_upper:.3f}  LP_approx={t_approx:.3f}  |  MILP LPs solved: {milp_num_lps}")

    return pd.concat([points.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
