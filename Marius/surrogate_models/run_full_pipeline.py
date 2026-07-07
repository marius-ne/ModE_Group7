"""
Meta-script: runs the full surrogate-model pipeline for five modes:
  - 1D_log:   ratio -> specific OPEX, trained on a log-spaced ratio sweep.
  - 1D_angle: ratio -> specific OPEX, trained on a ratio sweep equally spaced in
              arctan(ratio) instead of log(ratio) (see Erdem's create_sample("angle", ...)).
  - 2D:       (gas_price, electricity_price) -> absolute OPEX, trained on LHS price pairs.
  - 2D_noY:   same 2D price-pair training data, but the linear regression is forced
              through the origin (fit_intercept=False) — the theoretically consistent
              choice, since true OPEX(c_g, c_el) is homogeneous of degree 1 and should
              vanish at zero prices.
  - 1D_eqv:   same LHS price-pair training data as 2D, but converted per-sample to
              (ratio -> specific OPEX) via the specific_opex = absolute_opex / c_el
              identity before regression — no re-solving. Isolates the effect of
              "1D ratio feature" vs "2D price-pair feature" while holding the
              underlying training scenarios fixed.

1D_log and 1D_angle are trained and evaluated side by side (rather than picking one
via a single switch) so the effect of the ratio-sampling method is directly visible
in the R^2 comparison.

Test set: every mode is evaluated on the exact same underlying price scenarios.
ONE shared test set of n_test (gas, electricity) price pairs, drawn i.i.d.
uniformly at random within the feasible price rectangle, is generated and solved
once (evaluate_on_test_samples.run_shared); the 1D test set (also used by 1D_log,
1D_angle and 1D_eqv) is derived from it exactly (no re-solving) via the same
specific-OPEX identity.

Training data is otherwise per-source (1D_log/1D_angle: ratio sweeps; 2D: LHS price
pairs — 2D_noY and 1D_eqv both reuse 2D's training data, no re-solving), generated
via Marius/evaluation/evaluate_on_training_samples.py.

Then for each mode, train the 4 linear-regression surrogates and evaluate them
(train_surrogate_models.py), and finally compare train/test R^2 across modes and
formulations.

All outputs live under Marius/ (results/ and surrogate_models/{models,results}/{1D_log,1D_angle,2D,2D_noY,1D_eqv}/)
and never touch Florian/'s own surrogate model files.

Run from the repo root:  python Marius/surrogate_models/run_full_pipeline.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_on_training_samples
import evaluate_on_test_samples
import train_surrogate_models as trainer
from _evaluation_common import derive_1d_from_2d

MODES = ["1D_log", "1D_angle", "2D", "2D_noY", "1D_eqv"]

# How each mode's training data is generated. "1D_eqv" is handled specially in
# main() (derived from "2D"'s training data, no entry needed here).
TRAIN_SOURCE_OF = {
    "1D_log": dict(sampling_mode="1D", method_1d="log"),
    "1D_angle": dict(sampling_mode="1D", method_1d="angle"),
    "2D": dict(sampling_mode="2D"),
}
# Which training data (key into train_csv_of) each mode uses.
TRAIN_KEY_OF = {"1D_log": "1D_log", "1D_angle": "1D_angle", "2D": "2D", "2D_noY": "2D", "1D_eqv": "1D_eqv"}
# Which shared test set each mode is evaluated on.
TEST_KEY_OF = {"1D_log": "1D", "1D_angle": "1D", "2D": "2D", "2D_noY": "2D", "1D_eqv": "1D"}
FIT_INTERCEPT_OF = {"1D_log": True, "1D_angle": True, "2D": True, "2D_noY": False, "1D_eqv": True}

N_TRAIN = 40
N_TEST = 40  # size of the single shared test set used by every mode

# If True, reuse an existing evaluation_{training,test}_samples_<mode>.csv instead of
# re-solving the optimization problems, when both files are already present on disk.
REUSE_EXISTING_DATA = True

# Ground-truth units for compare_r2_vs_milp()'s 1D-family modes: "absolute" (€, rescaled
# by the real electricity price) or "specific" (each mode's own native €/(€/kWh) units).
MILP_COMPARISON_UNIT = "absolute"

OUT_DIR = Path("Marius/surrogate_models")
RESULTS_DIR = OUT_DIR / "results"
MODELS_DIR = OUT_DIR / "models"


Y_PAD_FRAC = 0.1  # fraction of the value range added as padding below/above the bars


def _set_ylim_with_padding(ax, all_train_r2: dict, all_test_r2: dict, pad_frac: float = Y_PAD_FRAC):
    """Set the (shared) y-axis limits to [min, max] of every plotted value, padded by
    pad_frac of the range, instead of matplotlib's default 0-anchored bar axis."""
    values = [
        r2_by_mode[col]
        for r2_by_mode in (*all_train_r2.values(), *all_test_r2.values())
        for col in trainer.OPEX_COLUMNS
    ]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * pad_frac if hi > lo else 0.01
    ax.set_ylim(lo - pad, hi + pad)


def compare_r2(all_train_r2: dict, all_test_r2: dict):
    """Save a comparison table + grouped bar chart of R^2 per formulation, mode and split."""
    rows = []
    for mode in MODES:
        for col in trainer.OPEX_COLUMNS:
            rows.append({"mode": mode, "formulation": col, "split": "train", "r2": all_train_r2[mode][col]})
            rows.append({"mode": mode, "formulation": col, "split": "test", "r2": all_test_r2[mode][col]})
    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "r2_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved R^2 comparison table to {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    n_modes = len(MODES)
    width = 0.8 / n_modes
    x = range(len(trainer.OPEX_COLUMNS))
    for ax, split in zip(axes, ["train", "test"]):
        for i, mode in enumerate(MODES):
            r2_by_mode = all_train_r2[mode] if split == "train" else all_test_r2[mode]
            values = [r2_by_mode[col] for col in trainer.OPEX_COLUMNS]
            offset = (i - (n_modes - 1) / 2) * width
            ax.bar([xi + offset for xi in x], values, width=width, label=mode)
        ax.set_xticks(list(x))
        ax.set_xticklabels(trainer.TITLES, rotation=15)
        ax.set_ylabel("$R^2$")
        ax.set_title(f"{split.capitalize()} set")
        ax.grid(True, axis="y", alpha=0.4)
        ax.legend(title="Mode (target)", fontsize=8)

    _set_ylim_with_padding(axes[0], all_train_r2, all_test_r2)

    plt.suptitle(
        "Surrogate model $R^2$ — 1D_log / 1D_angle (ratio → specific OPEX, different ratio sampling)  |  "
        "2D (price pair → absolute OPEX, with intercept)  |  "
        "2D_noY (same data, no intercept)  |  "
        "1D_eqv (2D's training data, converted to ratio → specific OPEX)",
        fontsize=10,
    )
    plt.tight_layout()
    plot_path = RESULTS_DIR / "r2_comparison.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R^2 comparison plot to {plot_path}")


def _r2_vs_milp(mode: str, df_eval: pd.DataFrame, df_2d_aligned: pd.DataFrame | None, unit: str) -> dict:
    """R^2 of each formulation's surrogate prediction on df_eval against the TRUE
    MILP OPEX, as either "absolute" (€) or "specific" (€/(€/kWh), each mode's native
    units) ground truth -- a single common comparison basis across every mode.

    2D-family predictions are already absolute €, used as-is regardless of unit.
    1D-family predictions are *specific* OPEX; when unit="absolute", df_2d_aligned
    (if given) is the row-for-row 2D counterpart those points were derived from (see
    derive_1d_from_2d) and supplies the REAL, per-point electricity price to convert
    both prediction and ground truth to absolute €. df_2d_aligned is None only for
    1D_log/1D_angle *training* data, a pure ratio sweep with no associated real price
    pair -- there, no conversion is possible, but R^2 is scale-invariant under any
    single fixed multiplier applied uniformly to truth and prediction, so the
    specific-OPEX R^2 already equals what any fixed reference c_el would give, and is
    used as-is regardless of unit.
    """
    is_2d = "gas_price_MWh" in df_eval.columns
    feature_cols = trainer.feature_cols_of(df_eval)
    r2 = {}
    for col in trainer.OPEX_COLUMNS:
        model = joblib.load(MODELS_DIR / mode / f"surrogate_{col}.joblib")
        pred = model.predict(df_eval[feature_cols])
        if is_2d or unit == "specific" or df_2d_aligned is None:
            y_true, pred_scored = df_eval["opex_milp"].to_numpy(), pred
        else:
            c_el = df_2d_aligned["electricity_price_MWh"].to_numpy() / 1000.0
            y_true, pred_scored = df_2d_aligned["opex_milp"].to_numpy(), pred * c_el
        r2[col] = r2_score(y_true, pred_scored)
    return r2


def compare_r2_vs_milp(train_csv_of: dict, test_csv_of: dict, unit: str = MILP_COMPARISON_UNIT):
    """Compare every mode's 4 surrogate predictions against the TRUE MILP OPEX as a
    single common ground truth -- i.e. how well does the LP_lower/LP_upper/LP_approx
    surrogate (and, as a reference point, the MILP surrogate itself) actually
    approximate the true optimal MILP cost?

    unit="absolute" (default): ground truth and 1D-family predictions are both
        converted to real € (see _r2_vs_milp) so every mode answers the same question.
    unit="specific": 1D-family modes are scored in their own native specific-OPEX
        units instead (2D-family is unaffected either way, it's already absolute).

    Uses only already-computed data: opex_milp is already a column in every mode's CSV
    (solve_all always solves all 4 formulations for the same points), the 1D CSVs are
    exact row-for-row derivations of a 2D CSV (so the real price needed for the
    specific -> absolute conversion is recoverable), and the 4 models per mode are
    already trained and saved to disk by main() -- no re-solving.
    """
    df_2d_train = pd.read_csv(train_csv_of["2D"])
    df_2d_test = pd.read_csv(test_csv_of["2D"])
    # mode -> (aligned 2D train df or None, aligned 2D test df or None)
    aligned_2d_of = {
        "1D_log": (None, df_2d_test),
        "1D_angle": (None, df_2d_test),
        "2D": (None, None),
        "2D_noY": (None, None),
        "1D_eqv": (df_2d_train, df_2d_test),
    }

    all_train_r2 = {}
    all_test_r2 = {}
    for mode in MODES:
        df_train = pd.read_csv(train_csv_of[TRAIN_KEY_OF[mode]])
        df_test = pd.read_csv(test_csv_of[TEST_KEY_OF[mode]])
        aligned_train, aligned_test = aligned_2d_of[mode]

        all_train_r2[mode] = _r2_vs_milp(mode, df_train, aligned_train, unit)
        all_test_r2[mode] = _r2_vs_milp(mode, df_test, aligned_test, unit)

    rows = []
    for mode in MODES:
        for col in trainer.OPEX_COLUMNS:
            rows.append({"mode": mode, "formulation": col, "split": "train", "r2_vs_milp": all_train_r2[mode][col]})
            rows.append({"mode": mode, "formulation": col, "split": "test", "r2_vs_milp": all_test_r2[mode][col]})
    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"r2_vs_milp_comparison_{unit}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved R^2-vs-MILP ({unit}) comparison table to {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    n_modes = len(MODES)
    width = 0.8 / n_modes
    x = range(len(trainer.OPEX_COLUMNS))
    ylabel = "$R^2$ vs true MILP [€]" if unit == "absolute" else "$R^2$ vs true MILP (native units)"
    for ax, split in zip(axes, ["train", "test"]):
        for i, mode in enumerate(MODES):
            r2_by_mode = all_train_r2[mode] if split == "train" else all_test_r2[mode]
            values = [r2_by_mode[col] for col in trainer.OPEX_COLUMNS]
            offset = (i - (n_modes - 1) / 2) * width
            ax.bar([xi + offset for xi in x], values, width=width, label=mode)
        ax.set_xticks(list(x))
        ax.set_xticklabels(trainer.TITLES, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{split.capitalize()} set")
        ax.grid(True, axis="y", alpha=0.4)
        ax.legend(title="Mode (target)", fontsize=8)

    _set_ylim_with_padding(axes[0], all_train_r2, all_test_r2)

    if unit == "absolute":
        subtitle = ("1D-family predictions rescaled by the real electricity price before scoring; "
                    "1D_log/1D_angle train has no real price pair, scored in its native scale-invariant units")
    else:
        subtitle = "1D-family modes scored in their own native specific-OPEX units (not converted to €)"
    plt.suptitle(
        f"Surrogate model $R^2$ relative to the true MILP ground truth ({unit})\n{subtitle}",
        fontsize=9,
    )
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"r2_vs_milp_comparison_{unit}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R^2-vs-MILP ({unit}) comparison plot to {plot_path}")


def main():
    all_train_r2 = {}
    all_test_r2 = {}

    # --- Shared test set: same underlying uniformly-random price scenarios for every mode ---
    test_csv_of = {
        "1D": Path("Marius/results/evaluation_test_samples_1D.csv"),
        "2D": Path("Marius/results/evaluation_test_samples_2D.csv"),
    }
    if REUSE_EXISTING_DATA and all(p.exists() for p in test_csv_of.values()):
        print(f"\n{'#' * 60}\n# Reusing existing shared test data\n{'#' * 60}")
        print("  " + "\n  ".join(str(p) for p in test_csv_of.values()))
    else:
        print(f"\n{'#' * 60}\n# Generating shared test set (n={N_TEST}, uniform random)\n{'#' * 60}")
        evaluate_on_test_samples.run_shared(
            N_TEST, out_csv_2d=test_csv_of["2D"], out_csv_1d=test_csv_of["1D"]
        )

    # --- Per-key training data (2D_noY reuses "2D"; 1D_eqv is derived from "2D", no re-solving) ---
    train_csv_of = {}  # train key -> train_csv
    for mode in MODES:
        train_key = TRAIN_KEY_OF[mode]

        if train_key not in train_csv_of:
            train_csv = Path(f"Marius/results/evaluation_training_samples_{train_key}.csv")

            if train_key == "1D_eqv":
                print(f"\n{'#' * 60}\n# Deriving 1D_eqv training data from 2D's training data\n{'#' * 60}")
                df_2d_train = pd.read_csv(train_csv_of["2D"])
                df_1d_eqv_train = derive_1d_from_2d(df_2d_train)
                train_csv.parent.mkdir(parents=True, exist_ok=True)
                df_1d_eqv_train.to_csv(train_csv, index=False)
                print(f"  Saved to {train_csv}")
            elif REUSE_EXISTING_DATA and train_csv.exists():
                print(f"\n{'#' * 60}\n# Reusing existing training data for: {train_key}\n{'#' * 60}")
                print(f"  {train_csv}")
            else:
                print(f"\n{'#' * 60}\n# Evaluating training samples for: {train_key}\n{'#' * 60}")
                evaluate_on_training_samples.run(
                    n_train=N_TRAIN, out_csv=train_csv, **TRAIN_SOURCE_OF[train_key]
                )

            train_csv_of[train_key] = train_csv

        print(f"\n{'#' * 60}\n# Training mode: {mode} (fit_intercept={FIT_INTERCEPT_OF[mode]})\n{'#' * 60}")
        train_r2, test_r2 = trainer.run(
            train_csv=train_csv_of[train_key], test_csv=test_csv_of[TEST_KEY_OF[mode]],
            models_dir=MODELS_DIR / mode, results_dir=RESULTS_DIR / mode,
            fit_intercept=FIT_INTERCEPT_OF[mode],
        )
        all_train_r2[mode] = train_r2
        all_test_r2[mode] = test_r2

    compare_r2(all_train_r2, all_test_r2)
    compare_r2_vs_milp(train_csv_of, test_csv_of)


if __name__ == "__main__":
    main()
