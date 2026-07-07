"""
Compare LP mean and MILP-regression predictions on the log-sampled test ratios.

For each ratio in Erdem/results/Sampling/test/log_10_samples.csv, this script
recomputes the LP lower and LP upper OPEX values. It then compares:

- mean(LP lower, LP upper)
- the saved regression model trained on MILP data

against the existing MILP OPEX values in Marius/results/opex_random_sample_10.csv
using R^2.
"""

import sys
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]
MARIUS_DIR = ROOT / "Marius"
sys.path.insert(0, str(MARIUS_DIR))

from formulation_LP_lower import solve as solve_lp_lower
from formulation_LP_upper import solve as solve_lp_upper


RATIOS_CSV = ROOT / "Erdem" / "results" / "Sampling" / "test" / "log_10_samples.csv"
MILP_RESULTS_CSV = ROOT / "Marius" / "results" / "opex_random_sample_10.csv"
MILP_REGRESSION_MODEL = ROOT / "Florian" / "surrogate_model_opex_milp.joblib"


def main():
    ratios = pd.read_csv(RATIOS_CSV)["ratios"].astype(float)
    df_milp_results = pd.read_csv(MILP_RESULTS_CSV)
    x_test = pd.DataFrame({"ratio": ratios})
    c_el = 1.0

    print(f"Loaded {len(ratios)} test samples from {RATIOS_CSV}")
    print(f"Loaded MILP actual values from {MILP_RESULTS_CSV}")

    if len(df_milp_results) != len(ratios):
        raise ValueError(
            f"Expected {len(ratios)} MILP result rows, but found {len(df_milp_results)}."
        )

    milp_actual = df_milp_results["opex_milp"].to_numpy()
    lp_mean_predictions = []

    print("\nComputing LP lower/upper mean predictions...")
    lp_mean_start = perf_counter()
    for i, ratio in enumerate(ratios, start=1):
        c_G = ratio * c_el
        opex_lower, _ = solve_lp_lower(c_G, c_el)
        opex_upper, _ = solve_lp_upper(c_G, c_el)
        lp_mean_predictions.append((opex_lower + opex_upper) / 2)
        print(
            f"[{i}/{len(ratios)}] ratio={ratio:.6f}  "
            f"LP mean prediction={lp_mean_predictions[-1]:,.2f}"
        )
    lp_mean_end = perf_counter()

    print("\nComputing MILP-regression predictions...")
    regression_total_start = perf_counter()
    model = joblib.load(MILP_REGRESSION_MODEL)
    regression_start = perf_counter()
    milp_regression_predictions = model.predict(x_test)
    regression_end = perf_counter()
    regression_total_end = perf_counter()

    r2_lp_mean = r2_score(milp_actual, lp_mean_predictions)
    r2_milp_regression = r2_score(milp_actual, milp_regression_predictions)

    print("\n=== R^2 compared to MILP OPEX from CSV ===")
    print(f"Mean of LP lower and LP upper : {r2_lp_mean:.6f}")
    print(f"MILP-trained regression       : {r2_milp_regression:.6f}")

    print("\n=== Runtime ===")
    print(f"LP lower/upper total time     : {lp_mean_end - lp_mean_start:.4f} s")
    print(f"MILP regression total time  (etails opening of model)  : {regression_total_end - regression_total_start:.6f} s")
    print(f"MILP regression prediction    : {regression_end - regression_start:.6f} s")


if __name__ == "__main__":
    main()
