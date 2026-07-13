"""
Train linear-regression surrogate models for the 4 canonical optimization
formulations in Erdem/src/optimization/core.py (MILP, LP lower, LP upper,
LP approximated).

Reads the already-solved training/test data produced by
Marius/evaluation/evaluate_on_training_samples.py and evaluate_on_test_samples.py
(features are either a "ratio" column for 1D sampling, or
"gas_price_MWh"/"electricity_price_MWh" columns for 2D sampling).

Note: for 1D sampling the opex_* targets are *specific* OPEX (€/(€/kWh)), not
absolute OPEX in € — see _evaluation_common.solve_all for why.

Steps:
  1. Load training data.
  2. Train one LinearRegression per formulation.
  3. Visualize actual-vs-predicted OPEX (with R^2) on the training set.
  4. Load test data.
  5. Visualize actual-vs-predicted OPEX (with R^2) on the test set.

Run from the repo root:  python Marius/surrogate_models/train_surrogate_models.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Erdem"))
from src.sampling.core import GAS_MIN, GAS_MAX, ELEC_MIN, ELEC_MAX

# ============================================================
# Config
# ============================================================
TRAIN_CSV = Path("Marius/results/evaluation_training_samples_1D.csv")
TEST_CSV = Path("Marius/results/evaluation_test_samples_1D.csv")

OUT_DIR = Path("Marius/surrogate_models")
RESULTS_DIR = OUT_DIR / "results"
MODELS_DIR = OUT_DIR / "models"

OPEX_COLUMNS = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]
TITLES = ["MILP", "LP Lower", "LP Upper", "LP Approx"]
COLORS = ["#2166AC", "#4DAC26", "#D6604D", "#35978F"]


def feature_cols_of(df: pd.DataFrame) -> list:
    """Infer the feature columns: ["ratio"] for 1D sampling, else the 2D price pair."""
    return ["ratio"] if "ratio" in df.columns else ["gas_price_MWh", "electricity_price_MWh"]


def sample_weight_of(ratio: pd.Series) -> np.ndarray:
    """Weight each ratio sample by (a representative real electricity price at that
    ratio, within the true 2D price rectangle)^2.

    For a given ratio r = c_g/c_el, only c_el in [max(ELEC_MIN, GAS_MIN/r),
    min(ELEC_MAX, GAS_MAX/r)] keeps (c_g, c_el) = (r*c_el, c_el) inside the price
    rectangle actually sampled in 2D (see GAS_MIN/GAS_MAX/ELEC_MIN/ELEC_MAX in
    Erdem/src/sampling/core.py); we take the geometric mean of that interval as the
    representative c_el(r), and square it, since absolute-OPEX error scales with
    c_el (see specific_opex -> absolute_opex conversion), so absolute squared error
    scales with c_el^2 -- weighting the fit by c_el(r)^2 makes ordinary least squares
    minimize (an estimate of) absolute-OPEX squared error instead of specific-OPEX
    squared error.
    """
    ratio = ratio.to_numpy()
    c_el_low = np.maximum(ELEC_MIN, GAS_MIN / ratio)
    c_el_high = np.minimum(ELEC_MAX, GAS_MAX / ratio)
    return c_el_low * c_el_high


def opex_label_of(feature_cols: list) -> str:
    """1D (ratio) sampling fixes c_el to an arbitrary reference, so its opex_* columns are
    *specific* OPEX (OPEX / c_el), not absolute OPEX in € — see _evaluation_common.solve_all.
    """
    return "Specific OPEX [€/(€/kWh)]" if feature_cols == ["ratio"] else "OPEX [€]"


def plot_actual_vs_predicted(df: pd.DataFrame, models: dict, feature_cols: list, title: str, out_path: Path) -> dict:
    """2x2 grid of actual-vs-predicted OPEX scatter plots (one per formulation), R^2 annotated.

    Returns a dict of R^2 scores keyed by opex column.
    """
    opex_label = opex_label_of(feature_cols)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    r2_values = {}
    for ax, col, model_title, color in zip(axes.flatten(), OPEX_COLUMNS, TITLES, COLORS):
        y_true = df[col].to_numpy()
        y_pred = models[col].predict(df[feature_cols])
        r2 = r2_score(y_true, y_pred)
        r2_values[col] = r2

        ax.scatter(y_true, y_pred, color=color, edgecolors="black", alpha=0.75, s=45)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal prediction")

        ax.text(
            0.04, 0.97, f"$R^2$ = {r2:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85),
        )
        ax.set_xlabel(f"Actual {opex_label}")
        ax.set_ylabel(f"Predicted {opex_label}")
        ax.set_title(model_title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")
    return r2_values


def run(train_csv: Path = TRAIN_CSV, test_csv: Path = TEST_CSV,
        models_dir: Path = MODELS_DIR, results_dir: Path = RESULTS_DIR,
        fit_intercept: bool = True, weighted: bool = False, test_tag: str = ""):
    """Train the 4 surrogate LinearRegressions on train_csv, evaluate on test_csv.

    fit_intercept=False forces the regression through the origin — the theoretically
    consistent choice for the 2D price-pair case, since true OPEX(c_g, c_el) is
    homogeneous of degree 1 (zero prices imply zero OPEX), so a nonzero intercept is
    a modeling artifact there. Defaults to True (sklearn's own default).

    weighted=True fits the (ratio -> specific OPEX) regression with sample_weight_of's
    weights instead of ordinary least squares, so the fit targets absolute-OPEX error
    at realistic prices rather than specific-OPEX error at the arbitrary reference
    c_el used to generate 1D training data. Only valid for 1D (ratio) training data.

    test_tag identifies the test set (size + sampling method) and is appended to the
    test scatter plot's filename, so evaluating against a different test set does not
    overwrite the previous one's plot. The train scatter is unaffected by the test set
    and keeps its fixed name.

    Saves models (joblib) to models_dir and actual-vs-predicted plots to results_dir.
    Returns (train_r2, test_r2), each a dict of R^2 scores keyed by opex column.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load training data -------------------------------------------------
    df_train = pd.read_csv(train_csv)
    feature_cols = feature_cols_of(df_train)
    if feature_cols != ["ratio"] and weighted:
        raise ValueError("weighted=True is only supported for 1D (ratio) training data.")
    opex_label = opex_label_of(feature_cols)
    sample_weight = sample_weight_of(df_train["ratio"]) if weighted else None
    print(f"Loaded {len(df_train)} training points from {train_csv} "
          f"(features: {feature_cols}, target: {opex_label}, fit_intercept={fit_intercept}, "
          f"weighted={weighted})")

    # --- 2. Train the linear regressions ----------------------------------------
    models = {}
    for col in OPEX_COLUMNS:
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(df_train[feature_cols], df_train[col], sample_weight=sample_weight)
        models[col] = model
        joblib.dump(model, models_dir / f"surrogate_{col}.joblib")
        print(f"  {col}: coef={model.coef_}, intercept={model.intercept_:.2f}")

    # --- 3. Visualize training R^2 -----------------------------------------------
    train_r2 = plot_actual_vs_predicted(
        df_train, models, feature_cols,
        f"Training set — Actual vs. Predicted {opex_label} (n={len(df_train)})",
        results_dir / "train_r2_scatter.png",
    )

    # --- 4. Load test data ---------------------------------------------------------
    df_test = pd.read_csv(test_csv)
    print(f"Loaded {len(df_test)} test points from {test_csv}")

    # --- 5. Visualize test R^2 --------------------------------------------------
    suffix = f"_{test_tag}" if test_tag else ""
    test_r2 = plot_actual_vs_predicted(
        df_test, models, feature_cols,
        f"Test set — Actual vs. Predicted {opex_label} (n={len(df_test)})",
        results_dir / f"test_r2_scatter{suffix}.png",
    )

    print("\nDone.")
    return train_r2, test_r2


if __name__ == "__main__":
    run()
