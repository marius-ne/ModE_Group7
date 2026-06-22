"""
Compare surrogate models (linear regressions from Florian/) against true
optimisation results for the held-out test set.  For each of the 4
formulations (MILP, LP Lower, LP Upper, LP Approx) one subplot shows:

  - the linear regression curve over an extended price-ratio range
  - true OPEX values for the test samples (solved on demand or read from cache)
  - surrogate-predicted OPEX for those same test samples

Run from the repo root:  python Marius/visualization/plot_surrogate_comparison.py

Cache: true test values are stored in Marius/results/test_samples_true_opex.csv
so subsequent runs skip the expensive solver calls.  Delete that file to force
a re-solve.
"""

import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REWRITE    = True
CACHE_PATH   = Path("Marius/results/test_samples_true_opex.csv")
EVAL_CSV     = Path("Marius/results/evaluation_log_samples.csv")
TRAINING_CSV = Path("Erdem/results/Sampling/log_training_samples.csv")

OPEX_COLUMNS = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]
TITLES       = ["MILP", "LP Lower", "LP Upper", "LP Approx"]
COLORS       = ["#2166AC", "#4DAC26", "#D6604D", "#35978F"]
MODEL_PATHS  = [
    "Florian/surrogate_model_opex_milp.joblib",
    "Florian/surrogate_model_opex_lp_lower.joblib",
    "Florian/surrogate_model_opex_lp_upper.joblib",
    "Florian/surrogate_model_opex_lp_approx.joblib",
]

# ---------------------------------------------------------------------------
# Reproduce the train/test split from Florian/surrogate_models/training_marius.py
# (test_size=0.2, random_state=42)
# ---------------------------------------------------------------------------
df_eval = pd.read_csv(EVAL_CSV)
X = df_eval[["ratio"]]
_, X_test, _, _ = train_test_split(X, df_eval[OPEX_COLUMNS[0]], test_size=0.2, random_state=42)
test_ratios = np.sort(X_test["ratio"].values)

# ---------------------------------------------------------------------------
# True test values — compute once, cache for future runs
# ---------------------------------------------------------------------------
if not REWRITE and CACHE_PATH.exists():
    print(f"Loading cached true test values from {CACHE_PATH}")
    test_true_df = pd.read_csv(CACHE_PATH)
else:
    print("Computing true OPEX for test samples (this may take a while)…")
    sys.path.append("Erdem")
    from src.optimization.core import solve_milp, solve_lp_lower, solve_lp_upper, solve_lp_approximated

    import pandas as _pd
    from pathlib import Path as _Path
    _demand_df = _pd.read_csv(_Path("energy_demands.csv"))
    _Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
    _P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()

    c_el = 1.0
    rows = []
    for i, ratio in enumerate(test_ratios):
        c_G = ratio * c_el
        print(f"  [{i+1}/{len(test_ratios)}] ratio={ratio:.6f}")
        opex_milp   = solve_milp(_Q_D, _P_D, c_G, c_el, mip_gap=1e-2)[0]
        opex_lower  = solve_lp_lower(_Q_D, _P_D, c_G, c_el)[0]
        opex_upper  = solve_lp_upper(_Q_D, _P_D, c_G, c_el)[0]
        opex_approx = solve_lp_approximated(_Q_D, _P_D, c_G, c_el, mode="mean_efficiency")[0]
        rows.append({
            "ratio":          ratio,
            "opex_milp":      opex_milp,
            "opex_lp_lower":  opex_lower,
            "opex_lp_upper":  opex_upper,
            "opex_lp_approx": opex_approx,
        })
        print(f"    MILP={opex_milp:,.0f}  LP_lower={opex_lower:,.0f}  "
              f"LP_upper={opex_upper:,.0f}  LP_approx={opex_approx:,.0f}")

    test_true_df = pd.DataFrame(rows)
    test_true_df.to_csv(CACHE_PATH, index=False)
    print(f"Saved to {CACHE_PATH}")

# ---------------------------------------------------------------------------
# Extended x-range for regression line (training range + ~10 % of log-span on each side)
# ---------------------------------------------------------------------------
training_ratios = pd.read_csv(TRAINING_CSV)["ratios"].values
log_min  = np.log(training_ratios.min())
log_max  = np.log(training_ratios.max())
log_span = log_max - log_min
ext      = 0.10  # fractional extension on each side

x_plot = np.logspace(
    np.log10(np.exp(log_min - ext * log_span)),
    np.log10(np.exp(log_max + ext * log_span)),
    300,
)

# ---------------------------------------------------------------------------
# Load surrogate models
# ---------------------------------------------------------------------------
models = []
for col, path in zip(OPEX_COLUMNS, MODEL_PATHS):
    m = joblib.load(path)
    print(f"\n--- {path} ---")
    print(f"  type       : {type(m)}")
    print(f"  coef_      : {m.coef_}")
    print(f"  intercept_ : {m.intercept_}")
    if hasattr(m, "feature_names_in_"):
        print(f"  features   : {m.feature_names_in_}")
    models.append(m)

# ---------------------------------------------------------------------------
# Plot — 2×2 grid, one subplot per formulation
# ---------------------------------------------------------------------------
OPEX_YLABEL = r"OPEX $\left[\dfrac{€_{\mathrm{OPEX}}}{€\,/\,\mathrm{kWh}}\right]$"
LEGEND_KW   = dict(framealpha=0.85, edgecolor="gray", fontsize=8)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, col, title, color, model in zip(
    axes.flatten(), OPEX_COLUMNS, TITLES, COLORS, models
):
    # Regression line over extended range
    y_line = model.predict(pd.DataFrame({"ratio": x_plot}))
    ax.plot(x_plot, y_line, color=color, linewidth=2.0, label="Surrogate (linear reg.)")

    # True test sample values
    true_vals = test_true_df[col].values
    ax.scatter(
        test_true_df["ratio"], true_vals,
        color="black", marker="o", s=50, zorder=6, label="True (test set)",
    )

    # Surrogate predictions at test sample locations
    pred_vals = model.predict(test_true_df[["ratio"]])
    ax.scatter(
        test_true_df["ratio"], pred_vals,
        color=color, marker="^", s=50, edgecolors="black", linewidths=0.7,
        zorder=6, label="Predicted (test set)",
    )

    # Vertical lines connecting true vs predicted for easy residual reading
    for r, t, p in zip(test_true_df["ratio"], true_vals, pred_vals):
        ax.plot([r, r], [t, p], color="gray", linewidth=0.8, alpha=0.55, zorder=5)

    # Training range boundaries
    ax.axvline(
        training_ratios.min(), color="gray", linewidth=0.9,
        linestyle="--", alpha=0.6, label="Training bounds",
    )
    ax.axvline(
        training_ratios.max(), color="gray", linewidth=0.9,
        linestyle="--", alpha=0.6,
    )

    # Rug marks at every training sample (blended: data-x, axes-y so height is scale-independent)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        np.repeat(training_ratios, 3),
        np.tile([0.0, 0.045, np.nan], len(training_ratios)),
        transform=trans, color="gray", linewidth=0.6, alpha=0.3,
        solid_capstyle="butt", label="Training samples",
    )

    # Metrics + equation annotation
    coef      = model.coef_[0]
    intercept = model.intercept_
    sign      = "+" if intercept >= 0 else "-"
    r2        = r2_score(true_vals, pred_vals)
    mse       = mean_squared_error(true_vals, pred_vals)
    ax.text(
        0.04, 0.97,
        f"$\\hat{{y}} = {float(coef):.0f}\\,r\\,{sign}\\,{abs(float(intercept)):.0f}$"        f"\n$R^2$ = {r2:.4f}\nRMSE = {np.sqrt(mse):,.0f}",
        transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Price ratio $c_G\,/\,c_{\mathrm{el}}$ $[-]$", fontsize=10)
    ax.set_ylabel(OPEX_YLABEL, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(**LEGEND_KW)
    ax.grid(True, which="both", ls="--", alpha=0.4)

plt.suptitle(
    "Surrogate Models vs True Optimisation — 4 Formulations\n"
    "(dashed lines = training data boundaries)",
    fontsize=13,
)
plt.tight_layout()

out_path = "Marius/visualization/surrogate_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Figure saved to {out_path}")
plt.show()
plt.close()
