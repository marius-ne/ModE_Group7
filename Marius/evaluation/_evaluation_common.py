"""
Shared helpers for evaluate_on_training_samples.py and evaluate_on_test_samples.py:
generating price sample points (2D price pairs or 1D price ratios) and solving
the 4 canonical optimization formulations in Erdem/src/optimization/core.py for
each point.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append("Erdem")
from src.sampling.core import create_sample
from src.optimization.core import solve_milp, solve_lp_lower, solve_lp_upper, solve_lp_approximated

MIP_GAP = 1e-2
STRICT_DEMAND_SATISFACTION = True
LP_APPROX_MODE = "mean_efficiency"
C_EL_REF = 1.0  # reference electricity price [€/kWh] used to derive c_gas = ratio * C_EL_REF in 1D mode

OPEX_COLUMNS = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]

_demand_df = pd.read_csv(Path("energy_demands.csv"))
Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()


def generate_points(sampling_mode: str, n: int, is_train: bool, method_2d: str = "sobol",
                     method_1d: str = "log") -> pd.DataFrame:
    """Generate n sample points as either (gas_price_MWh, electricity_price_MWh) pairs or price ratios.

    :param sampling_mode: "2D" for price pairs, "1D" for price ratios.
    :param is_train: training points use Sobol/LHS/log/angle; test points are drawn
        i.i.d. uniformly at random within the price rectangle (see generate_shared_test_points).
    :param method_2d: sampling method for 2D training mode ("sobol", "lhs" or "random").
    :param method_1d: sampling method for 1D training mode ("log" or "angle").
    """
    if sampling_mode == "2D":
        method = method_2d if is_train else "random"
        df, _ = create_sample(method, n)
        points = df[["gas_price", "electricity_price"]].reset_index(drop=True)
        return points.rename(columns={"gas_price": "gas_price_MWh", "electricity_price": "electricity_price_MWh"})
    elif sampling_mode == "1D":
        if is_train:
            ratios = create_sample(method_1d, n)
        else:
            df = generate_shared_test_points(n)
            ratios = df["gas_price_MWh"] / df["electricity_price_MWh"]
        return pd.DataFrame({"ratio": ratios.reset_index(drop=True)})
    else:
        raise ValueError(f"Unknown sampling_mode '{sampling_mode}', expected '1D' or '2D'.")


def generate_shared_test_points(n: int) -> pd.DataFrame:
    """Generate n (gas_price_MWh, electricity_price_MWh) pairs drawn i.i.d. uniformly at
    random within the feasible price rectangle — Erdem's create_sample("random", ...).

    This is the single test-point generator every sampling mode's test set is built
    from (see derive_1d_from_2d), so 1D, 2D and 2D_noY are all evaluated on the exact
    same underlying price scenarios.
    """
    df, _ = create_sample("random", n)
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
    return df_1d


def solve_all(points: pd.DataFrame) -> pd.DataFrame:
    """Solve all 4 optimization problems for every sample point, return points + opex columns.

    In 2D mode, c_g/c_el are the real sampled prices, so the returned opex_* columns are
    absolute OPEX in €. In 1D mode, c_el is pinned to the arbitrary reference C_EL_REF
    (not a real price), so the returned opex_* columns are actually *specific* OPEX
    (OPEX per unit electricity price, i.e. OPEX / c_el, units €/(€/kWh)) — to get the
    absolute OPEX for a real c_el, multiply the returned value by that real c_el.
    """
    is_2d = "gas_price_MWh" in points.columns
    opex_kind = "absolute OPEX [€]" if is_2d else "specific OPEX [€/(€/kWh)]"
    rows = []
    for i, row in points.iterrows():
        if is_2d:
            c_g = row["gas_price_MWh"] / 1000.0
            c_el = row["electricity_price_MWh"] / 1000.0
        else:
            c_el = C_EL_REF
            c_g = row["ratio"] * c_el

        print(f"  [{i + 1}/{len(points)}] c_g={c_g:.5f} €/kWh  c_el={c_el:.5f} €/kWh  ({opex_kind})")
        opex_milp = solve_milp(
            Q_D, P_D, c_g, c_el, mip_gap=MIP_GAP, strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )[0]
        opex_lower = solve_lp_lower(
            Q_D, P_D, c_g, c_el, strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )[0]
        opex_upper = solve_lp_upper(
            Q_D, P_D, c_g, c_el, strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )[0]
        opex_approx = solve_lp_approximated(
            Q_D, P_D, c_g, c_el, mode=LP_APPROX_MODE, strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )[0]

        rows.append({
            "opex_milp": opex_milp,
            "opex_lp_lower": opex_lower,
            "opex_lp_upper": opex_upper,
            "opex_lp_approx": opex_approx,
        })
        print(f"    MILP={opex_milp:,.2f}  LP_lower={opex_lower:,.2f}  "
              f"LP_upper={opex_upper:,.2f}  LP_approx={opex_approx:,.2f}")

    return pd.concat([points.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
