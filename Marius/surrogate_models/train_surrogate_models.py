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

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
        fit_intercept: bool = True):
    """Train the 4 surrogate LinearRegressions on train_csv, evaluate on test_csv.

    fit_intercept=False forces the regression through the origin — the theoretically
    consistent choice for the 2D price-pair case, since true OPEX(c_g, c_el) is
    homogeneous of degree 1 (zero prices imply zero OPEX), so a nonzero intercept is
    a modeling artifact there. Defaults to True (sklearn's own default).

    Saves models (joblib) to models_dir and actual-vs-predicted plots to results_dir.
    Returns (train_r2, test_r2), each a dict of R^2 scores keyed by opex column.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load training data -------------------------------------------------
    df_train = pd.read_csv(train_csv)
    feature_cols = feature_cols_of(df_train)
    opex_label = opex_label_of(feature_cols)
    print(f"Loaded {len(df_train)} training points from {train_csv} "
          f"(features: {feature_cols}, target: {opex_label}, fit_intercept={fit_intercept})")

    # --- 2. Train the linear regressions ----------------------------------------
    models = {}
    for col in OPEX_COLUMNS:
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(df_train[feature_cols], df_train[col])
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
    test_r2 = plot_actual_vs_predicted(
        df_test, models, feature_cols,
        f"Test set — Actual vs. Predicted {opex_label} (n={len(df_test)})",
        results_dir / "test_r2_scatter.png",
    )

    print("\nDone.")
    return train_r2, test_r2


if __name__ == "__main__":
    run()
