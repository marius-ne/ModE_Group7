"""
Meta-script: runs the full surrogate-model pipeline for six modes:
  - 1D_log:   ratio -> specific OPEX, trained on a log-spaced ratio sweep.
  - 1D_angle: ratio -> specific OPEX, trained on a ratio sweep equally spaced in
              arctan(ratio) instead of log(ratio) (see Erdem's create_sample("angle", ...)).
  - 1D_angle_weighted: same ratio sweep and specific-OPEX targets as 1D_angle (no
              re-solving), but fit with sample_weight_of's weights instead of OLS, so
              the fit minimizes (an estimate of) absolute-OPEX squared error at
              realistic prices rather than specific-OPEX squared error at the
              arbitrary reference c_el used to generate 1D training data.
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
1D_angle, 1D_angle_weighted and 1D_eqv) is derived from it exactly (no re-solving)
via the same specific-OPEX identity.

Training data is otherwise per-source (1D_log/1D_angle: ratio sweeps; 2D: LHS price
pairs — 2D_noY and 1D_eqv both reuse 2D's training data, no re-solving), generated
via Marius/evaluation/evaluate_on_training_samples.py.

Then for each mode, train the 4 linear-regression surrogates and evaluate them
(train_surrogate_models.py), and finally compare train/test R^2 across modes and
formulations.

All outputs live under Marius/ (results/ and
surrogate_models/{models,results}/{1D_log,1D_angle,1D_angle_weighted,2D,2D_noY,1D_eqv}/)
and never touch Florian/'s own surrogate model files.

Every result file whose contents depend on the test set (the R^2 comparison tables and
plots, and each mode's test scatter plot) has test_tag() -- "{N_TEST}_{method}_{bounds}_
seed{TEST_SEED}", e.g. "40_lhs_interior_only_seed4711" -- in its filename, so re-running with
a different test size, sampling method or seed adds new files instead of overwriting the
previous run's results.

Run from the repo root:  python Marius/surrogate_models/run_full_pipeline.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_on_training_samples
import train_surrogate_models as trainer
from _evaluation_common import derive_1d_from_2d, TEST_SEED

MODES = ["1D_log", "1D_angle", "1D_angle_weighted", "2D", "2D_noY", "1D_eqv"]

# How each mode's training data is generated. "1D_eqv" is handled specially in
# main() (derived from "2D"'s training data, no entry needed here).
# "1D_angle_weighted" reuses "1D_angle"'s training data (see TRAIN_KEY_OF), so it
# needs no entry here either.
TRAIN_SOURCE_OF = {
    "1D_log": dict(sampling_mode="1D", method_1d="log"),
    "1D_angle": dict(sampling_mode="1D", method_1d="angle"),
    "2D": dict(sampling_mode="2D"),
}
# Which training data (key into train_csv_of) each mode uses.
TRAIN_KEY_OF = {
    "1D_log": "1D_log", "1D_angle": "1D_angle", "1D_angle_weighted": "1D_angle",
    "2D": "2D", "2D_noY": "2D", "1D_eqv": "1D_eqv",
}
# Which test set each mode is evaluated on.
TEST_KEY_OF = {
    "1D_log": "1D", "1D_angle": "1D", "1D_angle_weighted": "1D",
    "2D": "2D", "2D_noY": "2D", "1D_eqv": "1D",
}
FIT_INTERCEPT_OF = {
    "1D_log": True, "1D_angle": True, "1D_angle_weighted": True,
    "2D": True, "2D_noY": False, "1D_eqv": True,
}
# Modes fit with sample_weight_of's weights instead of OLS (see train_surrogate_models.run).
WEIGHTED_OF = {"1D_angle_weighted": True}

N_TRAIN = 40
N_TEST = 10  # size of the single shared test set used by every mode
TEST_SAMPLING_METHOD_2D = "lhs"  # "lhs", "sobol" or "random"
TEST_N_CORNER = 0  # set to 0 to keep corners out of the test sample
TEST_N_EDGES = 0

# True  -> reuse the training/test CSVs already on disk and never re-solve. This is a
#          strict contract, not a best effort: the data for THIS config must already be
#          there, and every file is validated before use (exists, right number of points,
#          and the 1D/2D test pair really is one shared test set). Anything missing or
#          mismatched raises StaleDataError and stops the script -- it never quietly
#          falls back to solving, and never scores against another config's data.
# False -> regenerate EVERYTHING from scratch: the shared test set and all three training
#          sets are re-solved, 1D_eqv is re-derived, and every model, plot and comparison
#          table is rewritten. Nothing on disk is carried over.
REUSE_EXISTING_DATA = False

# Ground-truth units compare_r2_vs_milp() scores the 1D-family modes in. Both are always
# produced (they are pure post-processing of the same already-trained models, so computing
# both costs nothing and they answer different questions -- see compare_r2_vs_milp).
MILP_COMPARISON_UNITS = ["specific", "absolute"]

OUT_DIR = Path("Marius/surrogate_models")
RESULTS_DIR = OUT_DIR / "results"
MODELS_DIR = OUT_DIR / "models"


Y_PAD_FRAC = 0.1  # fraction of the value range added as padding below/above the bars


def boundary_tag() -> str:
    """How the price-rectangle boundary is treated in the test sample: "interior_only"
    means TEST_N_CORNER = TEST_N_EDGES = 0, i.e. no corner/edge-midpoint points are forced
    into the sample and every test point is a plain interior LHS draw. (It refers to those
    forced boundary points, not to the GAS_MIN/GAS_MAX/ELEC_MIN/ELEC_MAX price limits,
    which always apply.)"""
    return (
        "interior_only" if TEST_N_CORNER == 0 and TEST_N_EDGES == 0
        else f"{TEST_N_CORNER}corners_{TEST_N_EDGES}edges"
    )


def test_tag() -> str:
    """Identity of the test set (size, sampling method, boundary handling, seed), appended
    to every result filename that depends on it, so re-running with a different test set
    writes new files instead of overwriting the previous run's results.

    "random" carries no seed: create_sample's random branch draws from an unseeded
    np.random.uniform, so its points are not reproducible and a seed in the name would be
    a lie. The flip side is that consecutive "random" runs DO overwrite each other's
    results -- there is no stable identity to key them on.
    """
    method = TEST_SAMPLING_METHOD_2D.lower()
    seed_tag = "" if method == "random" else f"_seed{TEST_SEED}"
    return f"{N_TEST}_{method}_{boundary_tag()}{seed_tag}"


def default_generated_test_csv(sampling_mode: str) -> Path:
    return Path(f"Marius/results/evaluation_test_samples_{test_tag()}_{sampling_mode}.csv")


class StaleDataError(RuntimeError):
    """REUSE_EXISTING_DATA=True, but the data this config needs is missing or does not
    match the config. Raised instead of silently re-solving (which would contradict the
    flag) or, worse, scoring against data from a different config."""


def require_reusable_csv(path: Path, n_expected: int, what: str) -> pd.DataFrame:
    """Load a CSV that REUSE_EXISTING_DATA=True expects to already exist for the current
    config. Raise if it is missing or has the wrong number of rows.

    The filenames encode the config (N_TRAIN; N_TEST/method/boundary/seed via test_tag),
    so a missing file means this exact config has simply never been run -- there is
    nothing to reuse, and the honest answer is to stop rather than quietly solve it.
    """
    if not path.exists():
        raise StaleDataError(
            f"REUSE_EXISTING_DATA=True, but no {what} exists for the current config:\n"
            f"    {path}\n"
            f"is missing, i.e. this config (N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, "
            f"method={TEST_SAMPLING_METHOD_2D}, corners={TEST_N_CORNER}, edges={TEST_N_EDGES}, "
            f"seed={TEST_SEED}) has never been run.\n"
            f"Set REUSE_EXISTING_DATA=False to solve it from scratch (slow), or restore a "
            f"config whose data is already on disk."
        )
    df = pd.read_csv(path)
    if len(df) != n_expected:
        raise StaleDataError(
            f"REUSE_EXISTING_DATA=True, but the existing {what}\n"
            f"    {path}\n"
            f"has {len(df)} rows while the config asks for {n_expected}. Refusing to reuse it: "
            f"the results would be labelled with this config but computed on another one.\n"
            f"Set REUSE_EXISTING_DATA=False to regenerate, or delete the mismatched file."
        )
    return df


def require_aligned_test_pair(df_1d: pd.DataFrame, df_2d: pd.DataFrame,
                              path_1d: Path, path_2d: Path) -> None:
    """Verify the reused 1D test CSV really is derive_1d_from_2d's exact row-for-row
    rewrite of the reused 2D one (same price points, specific = absolute / c_el).

    Every mode is supposed to be scored on ONE shared test set; if these two files drifted
    apart (e.g. one was regenerated on its own, or they were solved independently), the 1D
    and 2D modes would silently be compared on different data.
    """
    expected = derive_1d_from_2d(df_2d)
    for col in ["ratio", *trainer.OPEX_COLUMNS]:
        if not np.allclose(df_1d[col], expected[col], rtol=1e-6):
            raise StaleDataError(
                f"REUSE_EXISTING_DATA=True, but the 1D and 2D test sets are not the same "
                f"underlying test set:\n"
                f"    {path_1d}\n    {path_2d}\n"
                f"column '{col}' of the 1D file does not match derive_1d_from_2d(2D file) "
                f"(max rel. deviation "
                f"{np.max(np.abs(df_1d[col] - expected[col]) / np.abs(expected[col])):.2%}).\n"
                f"The two must come from a single run_shared call. Set "
                f"REUSE_EXISTING_DATA=False to regenerate both together."
            )


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
    csv_path = RESULTS_DIR / f"r2_comparison_{test_tag()}.csv"
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
        "1D_angle_weighted (same data, fit weighted by price-rectangle c_el(ratio)^2)  |  "
        "2D (price pair → absolute OPEX, no intercept)  |  "
        "2D_noY (same data, no intercept)  |  "
        "1D_eqv (2D's training data, converted to ratio → specific OPEX)",
        fontsize=9,
    )
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"r2_comparison_{test_tag()}.png"
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


def compare_r2_vs_milp(train_csv_of: dict, test_csv_of: dict, unit: str):
    """Compare every mode's 4 surrogate predictions against the TRUE MILP OPEX as a
    single common ground truth -- i.e. how well does the LP_lower/LP_upper/LP_approx
    surrogate (and, as a reference point, the MILP surrogate itself) actually
    approximate the true optimal MILP cost?

    unit="absolute": ground truth and 1D-family predictions are both converted to real €
        (see _r2_vs_milp) so every mode answers the same question. This is the only
        like-for-like 1D-vs-2D comparison; the specific-OPEX R^2 of a 1D mode and the
        absolute-OPEX R^2 of a 2D mode have different targets and different SST
        denominators, so they are not comparable to each other.
    unit="specific": 1D-family modes are scored in their own native specific-OPEX
        units instead (2D-family is unaffected either way, it's already absolute).

    main() runs both (MILP_COMPARISON_UNITS).

    Uses only already-computed data: opex_milp is already a column in every mode's CSV
    (solve_all always solves all 4 formulations for the same points), the 1D CSVs are
    exact row-for-row derivations of a 2D CSV (so the real price needed for the
    specific -> absolute conversion is recoverable), and the 4 models per mode are
    already trained and saved to disk by main() -- no re-solving.
    """
    df_2d_train = pd.read_csv(train_csv_of["2D"])
    df_1d_test = pd.read_csv(test_csv_of["1D"])
    df_2d_test = pd.read_csv(test_csv_of["2D"])
    aligned_2d_test = df_2d_test if len(df_2d_test) == len(df_1d_test) else None
    # mode -> (aligned 2D train df or None, aligned 2D test df or None)
    aligned_2d_of = {
        "1D_log": (None, aligned_2d_test),
        "1D_angle": (None, aligned_2d_test),
        "1D_angle_weighted": (None, aligned_2d_test),
        "2D": (None, None),
        "2D_noY": (None, None),
        "1D_eqv": (df_2d_train, aligned_2d_test),
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
    csv_path = RESULTS_DIR / f"r2_vs_milp_comparison_{unit}_{test_tag()}.csv"
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
    plot_path = RESULTS_DIR / f"r2_vs_milp_comparison_{unit}_{test_tag()}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R^2-vs-MILP ({unit}) comparison plot to {plot_path}")


def main():
    all_train_r2 = {}
    all_test_r2 = {}

    # --- Test set ----------------------------------------------------------------
    # ONE shared test set for every mode, always at the canonical test_tag()-stamped path.
    # The 1D and 2D CSVs are the two views of that one set (2D = the solved price pairs,
    # 1D = derive_1d_from_2d's exact row-for-row ratio/specific-OPEX rewrite of it), so
    # they are only ever generated together, by a single run_shared call. Reuse is keyed
    # on that path, whose tag encodes N_TEST, the sampling method, the boundary handling
    # and TEST_SEED -- so a config change can never silently score against a stale test
    # set, it just generates the new one.
    one_d_test_csv = default_generated_test_csv("1D")
    two_d_test_csv = default_generated_test_csv("2D")

    if REUSE_EXISTING_DATA:
        print(f"\n{'#' * 60}\n# Reusing existing shared test data\n{'#' * 60}")
        df_2d_test = require_reusable_csv(two_d_test_csv, N_TEST, "2D test set")
        df_1d_test = require_reusable_csv(one_d_test_csv, N_TEST, "1D test set")
        require_aligned_test_pair(df_1d_test, df_2d_test, one_d_test_csv, two_d_test_csv)
        print(f"  {two_d_test_csv}\n  {one_d_test_csv}")
        print(f"  OK: {N_TEST} points each, 1D is an exact derivation of 2D")
    else:
        print(
            f"\n{'#' * 60}\n"
            f"# Generating shared test set "
            f"(n={N_TEST}, {TEST_SAMPLING_METHOD_2D}, corners={TEST_N_CORNER}, "
            f"edges={TEST_N_EDGES}, seed={TEST_SEED})\n"
            f"{'#' * 60}"
        )
        import evaluate_on_test_samples

        evaluate_on_test_samples.run_shared(
            N_TEST,
            out_csv_2d=two_d_test_csv,
            out_csv_1d=one_d_test_csv,
            test_method_2d=TEST_SAMPLING_METHOD_2D,
            n_corner=TEST_N_CORNER,
            n_edges=TEST_N_EDGES,
        )

    test_csv_of = {
        "1D": one_d_test_csv,
        "2D": two_d_test_csv,
    }

    # --- Per-key training data (2D_noY reuses "2D"; 1D_eqv is derived from "2D", no re-solving) ---
    train_csv_of = {}  # train key -> train_csv
    for mode in MODES:
        train_key = TRAIN_KEY_OF[mode]

        if train_key not in train_csv_of:
            # N_TRAIN is in the name (train_key already carries the sampling method), so
            # changing N_TRAIN generates a new file instead of silently reusing a training
            # set of the wrong size. The old untagged evaluation_training_samples_*.csv are
            # left alone -- Florian/'s scripts read them by that exact name.
            train_csv = Path(f"Marius/results/evaluation_{N_TRAIN}_training_samples_{train_key}.csv")

            if train_key == "1D_eqv":
                print(f"\n{'#' * 60}\n# Deriving 1D_eqv training data from 2D's training data\n{'#' * 60}")
                df_2d_train = pd.read_csv(train_csv_of["2D"])
                df_1d_eqv_train = derive_1d_from_2d(df_2d_train)
                train_csv.parent.mkdir(parents=True, exist_ok=True)
                df_1d_eqv_train.to_csv(train_csv, index=False)
                print(f"  Saved to {train_csv}")
            elif REUSE_EXISTING_DATA:
                print(f"\n{'#' * 60}\n# Reusing existing training data for: {train_key}\n{'#' * 60}")
                require_reusable_csv(train_csv, N_TRAIN, f"'{train_key}' training set")
                print(f"  {train_csv}\n  OK: {N_TRAIN} points")
            else:
                print(f"\n{'#' * 60}\n# Evaluating training samples for: {train_key}\n{'#' * 60}")
                evaluate_on_training_samples.run(
                    n_train=N_TRAIN, out_csv=train_csv, **TRAIN_SOURCE_OF[train_key]
                )

            train_csv_of[train_key] = train_csv

        print(
            f"\n{'#' * 60}\n"
            f"# Training mode: {mode} (fit_intercept={FIT_INTERCEPT_OF[mode]}, "
            f"weighted={WEIGHTED_OF.get(mode, False)})\n{'#' * 60}"
        )
        train_r2, test_r2 = trainer.run(
            train_csv=train_csv_of[train_key], test_csv=test_csv_of[TEST_KEY_OF[mode]],
            models_dir=MODELS_DIR / mode, results_dir=RESULTS_DIR / mode,
            fit_intercept=FIT_INTERCEPT_OF[mode], weighted=WEIGHTED_OF.get(mode, False),
            test_tag=test_tag(),
        )
        all_train_r2[mode] = train_r2
        all_test_r2[mode] = test_r2

    compare_r2(all_train_r2, all_test_r2)
    for unit in MILP_COMPARISON_UNITS:
        compare_r2_vs_milp(train_csv_of, test_csv_of, unit=unit)


if __name__ == "__main__":
    main()
