"""
Simple evaluation script: for each surrogate mode/formulation trained by
run_full_pipeline.py, compute the max absolute deviation between the
surrogate's prediction and its ground-truth optimization formulation
(opex_milp / opex_lp_lower / opex_lp_upper / opex_lp_approx) on the shared
test set, and print it to the terminal.

Uses the models and test set already on disk under Marius/surrogate_models/
(models/<mode>/surrogate_<col>.joblib and the test_tag()-stamped test CSVs) --
no training or solving happens here.

Run from the repo root:  python Marius/surrogate_models/evaluate_max_deviation.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_surrogate_models as trainer
from run_full_pipeline import MODES, TEST_KEY_OF, MODELS_DIR, default_generated_test_csv


def main():
    test_csv_of = {
        "1D": default_generated_test_csv("1D"),
        "2D": default_generated_test_csv("2D"),
    }

    for mode in MODES:
        df_test = pd.read_csv(test_csv_of[TEST_KEY_OF[mode]])
        feature_cols = trainer.feature_cols_of(df_test)
        print(f"\n{mode} (n={len(df_test)}):")
        for col in trainer.OPEX_COLUMNS:
            model = joblib.load(MODELS_DIR / mode / f"surrogate_{col}.joblib")
            pred = model.predict(df_test[feature_cols])
            max_dev = np.max(np.abs(pred - df_test[col].to_numpy()))
            print(f"  {col}: max deviation = {max_dev:.4f}")

    # Ground-truth (not surrogate-dependent) LP_upper/LP_lower spread relative to MILP.
    # The ratio (LP_upper - LP_lower) / MILP is the same for the 1D and 2D test CSVs (both
    # are the same underlying points, just uniformly rescaled by c_el per row), so either
    # suffices -- the 2D one is used since it holds the real, unscaled OPEX in euros.
    df_2d_test = pd.read_csv(test_csv_of["2D"])
    spread_pct = (df_2d_test["opex_lp_upper"] - df_2d_test["opex_lp_lower"]).abs() / df_2d_test["opex_milp"].abs() * 100
    print(f"\nMax LP_upper-LP_lower spread on test set: {spread_pct.max():.2f}% of MILP")
    print(f"Mean LP_upper-LP_lower spread on test set: {spread_pct.mean():.2f}% of MILP")


if __name__ == "__main__":
    main()
