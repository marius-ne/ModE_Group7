"""
Compare LP mean and MILP-regression predictions on the log-sampled test ratios.

For each gas/electricity price pair in Erdem/results/Sampling/test/random_10_samples.csv, this script
calculates the corresponding ratio and
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
MARIUS_DIR = ROOT / "Marius" / "OUTDATED"
sys.path.insert(0, str(MARIUS_DIR))

from formulation_LP_lower import solve as solve_lp_lower
from formulation_LP_upper import solve as solve_lp_upper


RATIOS_CSV = ROOT / "Erdem" / "results" / "Sampling" / "test" / "random_10_samples.csv"
MILP_RESULTS_CSV = ROOT / "Marius" / "results" / "opex_random_sample_10.csv"
MILP_REGRESSION_MODEL = ROOT / "Florian" / "surrogate_model_opex_milp.joblib"
N_RUNS = 10


def main():
    samples = pd.read_csv(RATIOS_CSV)
    ratios = samples["gas_price"].astype(float) / samples["electricity_price"].astype(float)
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
    lp_mean_predictions = None
    lp_mean_times = []

    print("\nComputing LP lower/upper mean predictions...")
    for run in range(1, N_RUNS + 1):
        current_lp_mean_predictions = []
        lp_mean_start = perf_counter()
        for i, ratio in enumerate(ratios, start=1):
            c_G = ratio * c_el
            opex_lower, _ = solve_lp_lower(c_G, c_el)
            opex_upper, _ = solve_lp_upper(c_G, c_el)
            current_lp_mean_predictions.append((opex_lower + opex_upper) / 2)
            if run == 1:
                print(
                    f"[{i}/{len(ratios)}] ratio={ratio:.6f}  "
                    f"LP mean prediction={current_lp_mean_predictions[-1]:,.2f}"
                )
        lp_mean_end = perf_counter()
        lp_mean_times.append(lp_mean_end - lp_mean_start)
        if lp_mean_predictions is None:
            lp_mean_predictions = current_lp_mean_predictions
        print(f"LP run {run}/{N_RUNS}: {lp_mean_times[-1]:.4f} s")

    print("\nComputing MILP-regression predictions...")
    milp_regression_predictions = None
    regression_total_times = []
    regression_prediction_times = []
    for run in range(1, N_RUNS + 1):
        regression_total_start = perf_counter()
        model = joblib.load(MILP_REGRESSION_MODEL)
        regression_start = perf_counter()
        current_milp_regression_predictions = model.predict(x_test)
        regression_end = perf_counter()
        regression_total_end = perf_counter()

        regression_total_times.append(regression_total_end - regression_total_start)
        regression_prediction_times.append(regression_end - regression_start)
        if milp_regression_predictions is None:
            milp_regression_predictions = current_milp_regression_predictions
        print(
            f"Regression run {run}/{N_RUNS}: "
            f"total={regression_total_times[-1]:.6f} s, "
            f"prediction={regression_prediction_times[-1]:.6f} s"
        )

    r2_lp_mean = r2_score(milp_actual, lp_mean_predictions)
    r2_milp_regression = r2_score(milp_actual, milp_regression_predictions)

    print("\n=== R^2 compared to MILP OPEX from CSV ===")
    print(f"Mean of LP lower and LP upper : {r2_lp_mean:.6f}")
    print(f"MILP-trained regression       : {r2_milp_regression:.6f}")

    print("\n=== Runtime ===")
    print(f"Number of runs                : {N_RUNS}")
    print(f"LP lower/upper mean time      : {sum(lp_mean_times) / len(lp_mean_times):.4f} s")
    print(f"MILP regression mean total time (including model load): {sum(regression_total_times) / len(regression_total_times):.6f} s")
    print(f"MILP regression mean prediction time                  : {sum(regression_prediction_times) / len(regression_prediction_times):.6f} s")
    
    df_results = pd.DataFrame([
        {"method" : "lp_mean",
         "r2" : r2_lp_mean,
         "mean_runtime_s" : sum(lp_mean_times) / len(lp_mean_times)},
        {"method" : "milp_regression",
         "r2" : r2_milp_regression,
         "mean_runtime_s" : sum(regression_total_times) / len(regression_total_times)}
    ])
    results_csv = ROOT / "Florian" / "results" / "regression"
    results_csv.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_csv / "r2_comparison_lp_mean_vs_milp_regression.csv", index=False)
    print(f"\nSaved R^2 comparison results to {results_csv / 'r2_comparison_lp_mean_vs_milp_regression.csv'}")

if __name__ == "__main__":
    main()
