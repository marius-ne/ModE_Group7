"""
Pareto front of accuracy vs. compute time for the ways of getting an OPEX number.

This script is self-contained: it generates the shared test set itself, solves every
formulation on it here (timing each solve as it goes) and only *reads* the already-trained
linear-regression surrogates from Marius/surrogate_models/models/. It never trains anything,
and it never borrows an OPEX or a runtime from the pipeline's tables -- the times on the
x-axis are all measured by this script, on the same machine, on the same price points.

The solves are the expensive part (the gold MILP most of all: a 1e-3 gap is far more work
than the 1e-2 one), so they are cached. A run that solves saves its solved test set to
Marius/results/pareto_solves_<test tag>_gold<gap>_loose<gap>.csv, and later runs reuse it
instead of re-solving -- so iterating on the plot itself is instant. The filename carries the
test set's identity and both MIP gaps, and the file is re-validated (row count, columns, and
that its price points really are this config's) before being trusted, so a reused file can
never answer a different question than the one being asked. Set REUSE_EXISTING_SOLVES=False
to force a fresh solve. Reusing means the x-axis shows the times of the run that did the
solving -- fine on one machine, misleading across machines.

Ground truth is the MILP solved to a tight MIP gap (GOLD_MIP_GAP = 1e-3): the gold standard
every other method -- including the loose-gap MILP the surrogates were trained on -- is
scored against. Accuracy is R^2 against that gold MILP OPEX in absolute €, the only basis on
which the 1D (specific-OPEX) and 2D (absolute-OPEX) surrogates are comparable to each other
and to the LP bounds. The points on the front are:

  1. MILP (gold)   -- the ground truth itself, so R^2 = 1 by construction. Slowest.
  2. MILP (loose)  -- the same MILP at the pipeline's MIP_GAP (1e-2), i.e. what buying a
                      looser optimality tolerance actually costs in accuracy and saves in time.
  3. LP mean       -- mean(LP lower, LP upper). The two bounds bracket the MILP optimum,
                      so their midpoint is a cheap estimate of it. Costs two LP solves.
  4/5. LR 1D, LR 2D -- the two trained linear-regression surrogate families (LR_MODES): the
                      1D one maps the price ratio to specific OPEX, the 2D one maps the price
                      pair to absolute OPEX. Each family was trained against all four
                      formulations' OPEX (MILP, LP lower, LP upper, LP approx), and the one
                      shown here is whichever of those four best predicts the gold MILP --
                      chosen per family, so the two are picked by the same rule. Costs one
                      dot product.

The LR is timed on inference only (model.predict), the per-query cost. Training it is a
one-off offline cost that is amortized away over repeated queries, so it does not belong on
a per-query axis; the script prints it separately for reference, from the training set's
recorded MILP solve times if that CSV is around.

Run from the repo root:  python Marius/visualization/plot_pareto_accuracy_vs_time.py
"""

import sys
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Erdem"))
sys.path.insert(0, str(ROOT / "Marius" / "evaluation"))
sys.path.insert(0, str(ROOT / "Marius" / "surrogate_models"))

import run_full_pipeline as pipeline
import train_surrogate_models as trainer
from _evaluation_common import (
    MIP_GAP, STRICT_DEMAND_SATISFACTION, P_D, Q_D, _timed, generate_shared_test_points,
)
from src.optimization.core import solve_lp_lower, solve_lp_upper, solve_milp
from src.visualization.style import apply_style

# ============================================================
# Config
# ============================================================
# The test set this front is built on, pinned here rather than inherited from whatever
# run_full_pipeline's config happens to say at the moment. The Pareto front is a headline
# result, so it should not silently change basis because someone was mid-experiment with a
# different N_TEST in the pipeline. The points are the same shared test points the pipeline
# uses (same generator, same TEST_SEED), they are just re-solved here rather than read back.
N_TEST = 10
TEST_SAMPLING_METHOD_2D = "lhs"

# The MIP gap of the gold-standard MILP: the ground truth everything else is scored against.
# Tighter than _evaluation_common.MIP_GAP (1e-2), which is the gap the surrogates' training
# data was solved at and which appears on the front as its own, cheaper-but-looser method.
GOLD_MIP_GAP = 1e-3

# True  -> if this exact configuration's solves are already on disk (see solves_csv), read
#          them back instead of re-solving. The solves are the slow part of this script, so
#          this is what makes re-styling the plot cheap. NOTE that the x-axis then shows the
#          times measured by THAT run, on THAT machine -- reusing across machines compares
#          this plot's compute times to hardware they were not measured on.
# False -> always re-solve, and overwrite the saved solves with the fresh ones.
# Either way, a run that solves saves its results, so the next run can reuse them.
REUSE_EXISTING_SOLVES = True

# predict() on a handful of rows is microseconds, far below clock resolution, so time a
# batch of repeats and divide.
PREDICT_REPEATS = 1000

OUT_DIR = ROOT / "Marius" / "visualization"
SOLVES_DIR = ROOT / "Marius" / "results"

# Everything the front is built on except the LR inference, which is timed fresh every run
# because it costs microseconds.
SOLVE_COLUMNS = [
    "opex_milp_gold", "opex_milp", "opex_lp_lower", "opex_lp_upper",
    "time_milp_gold", "time_milp", "time_lp_lower", "time_lp_upper",
]

# How each formulation a surrogate can be trained against is written on the plot. The raw
# column names (opex_lp_upper, ...) are how the pipeline keys its models, not how the
# formulations are named in the report.
FORMULATION_LABELS = {
    "opex_milp": "MILP",
    "opex_lp_lower": "$LP^L$",
    "opex_lp_upper": "$LP^U$",
    "opex_lp_approx": "$LP^{approx}$",
}

# The two surrogate families on the front, and the name each goes by here. Only these two of
# run_full_pipeline's MODES are considered: they are the headline 1D (ratio -> specific OPEX,
# trained on the arctan-spaced ratio sweep) and 2D (price pair -> absolute OPEX) surrogates.
# The pipeline's other modes (1D_log, 1D_angle_weighted, 2D_noY, 1D_eqv) are ablations of
# these two and are left to the pipeline's own R^2 tables.
LR_MODES = {"1D_angle": "1D", "2D": "2D"}

GOLD_METHOD = f"MILP (gap {GOLD_MIP_GAP:g})"
LOOSE_METHOD = f"MILP (gap {MIP_GAP:g})"
LP_MEAN_METHOD = (f"mean({FORMULATION_LABELS['opex_lp_lower']}, "
                  f"{FORMULATION_LABELS['opex_lp_upper']})")

COLORS = {
    GOLD_METHOD: "#2166AC",
    LOOSE_METHOD: "#35978F",
    LP_MEAN_METHOD: "#D6604D",
    "Best 2D LR": "#4DAC26",
    "Best 1D LR": "#9970AB",
}

# The Pareto front line: its own colour, so it does not read as belonging to any one method.
FRONT_COLOR = "#E08214"


def pin_pipeline_test_set() -> None:
    """Point the pipeline's tag helper at THIS script's test set, so the output filenames
    keep using the pipeline's naming convention (defined in exactly one place) while this
    script still chooses which test set it is about."""
    pipeline.N_TEST = N_TEST
    pipeline.TEST_SAMPLING_METHOD_2D = TEST_SAMPLING_METHOD_2D


def solves_csv() -> Path:
    """Where this configuration's solves are cached.

    The name carries everything that determines the numbers inside: the test set's identity
    (pipeline.test_tag() -- size, sampling method, boundary handling, seed) and BOTH MIP gaps.
    So a run with a different gold gap, a different loose gap or a different test set writes
    its own file rather than silently reusing solves that answer a different question.
    """
    return SOLVES_DIR / (f"pareto_solves_{pipeline.test_tag()}"
                         f"_gold{GOLD_MIP_GAP:g}_loose{MIP_GAP:g}.csv")


def reusable_solves(points: pd.DataFrame) -> pd.DataFrame | None:
    """The cached solves for exactly these test points, or None if there is nothing usable.

    Being on disk under the right name is not enough: the file is only accepted if it has one
    row per test point, carries every SOLVE_COLUMN, and its price pairs really are the points
    generated for this config. Otherwise we would be plotting one configuration's accuracy
    under another's label -- better to re-solve.
    """
    path = solves_csv()
    if not (REUSE_EXISTING_SOLVES and path.exists()):
        return None

    df = pd.read_csv(path)
    missing = [col for col in SOLVE_COLUMNS if col not in df.columns]
    if missing:
        print(f"Ignoring {path}: missing columns {missing}. Re-solving.")
        return None
    if len(df) != len(points):
        print(f"Ignoring {path}: {len(df)} rows, expected {len(points)}. Re-solving.")
        return None
    if not np.allclose(df[points.columns].to_numpy(), points.to_numpy()):
        print(f"Ignoring {path}: its price points are not the ones this config generates. "
              f"Re-solving.")
        return None

    print(f"Reusing the solves in {path} (times are from THAT run, on THAT machine)")
    return df


def solve_test_points(points: pd.DataFrame) -> pd.DataFrame:
    """Solve every formulation on the given test points, timing each solve as it is made.

    Returns the price pairs plus, per point: the gold MILP OPEX (GOLD_MIP_GAP), the loose
    MILP OPEX (_evaluation_common.MIP_GAP) and both LP bounds, each with the wall-clock
    seconds its solve took. All OPEX columns are absolute € (the prices are the real sampled
    ones). The surrogates' training data is not touched.
    """
    print(f"Solving {len(points)} test points (nothing reusable on disk)")

    rows = []
    for i, row in points.iterrows():
        c_g = row["gas_price_MWh"] / 1000.0
        c_el = row["electricity_price_MWh"] / 1000.0
        print(f"  [{i + 1}/{len(points)}] c_g={c_g:.5f} €/kWh  c_el={c_el:.5f} €/kWh")

        opex_gold, t_gold = _timed(
            solve_milp, Q_D, P_D, c_g, c_el,
            mip_gap=GOLD_MIP_GAP, strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )
        opex_milp, t_milp = _timed(
            solve_milp, Q_D, P_D, c_g, c_el,
            mip_gap=MIP_GAP, strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )
        opex_lower, t_lower = _timed(
            solve_lp_lower, Q_D, P_D, c_g, c_el,
            strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )
        opex_upper, t_upper = _timed(
            solve_lp_upper, Q_D, P_D, c_g, c_el,
            strict_demand_satisfaction=STRICT_DEMAND_SATISFACTION
        )

        rows.append({
            "opex_milp_gold": opex_gold, "opex_milp": opex_milp,
            "opex_lp_lower": opex_lower, "opex_lp_upper": opex_upper,
            "time_milp_gold": t_gold, "time_milp": t_milp,
            "time_lp_lower": t_lower, "time_lp_upper": t_upper,
        })
        print(f"    MILP_gold={opex_gold:,.2f}  MILP={opex_milp:,.2f}  "
              f"LP_lower={opex_lower:,.2f}  LP_upper={opex_upper:,.2f}")
        print(f"    times [s]: MILP_gold={t_gold:.3f}  MILP={t_milp:.3f}  "
              f"LP_lower={t_lower:.3f}  LP_upper={t_upper:.3f}")

    return pd.concat([points.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def test_solves() -> pd.DataFrame:
    """The solved test set this front is built on: reused from disk if this exact config has
    already been solved, otherwise solved now and saved so the next run can reuse it.

    The solves are what makes this script slow -- the gold MILP most of all, since a 1e-3 gap
    is far more work than the 1e-2 one -- while the LR side is microseconds. Caching them
    means iterating on the plot itself does not pay for the solver again.
    """
    points = generate_shared_test_points(N_TEST, method_2d=TEST_SAMPLING_METHOD_2D)
    print(f"Test set ({pipeline.test_tag()}): {len(points)} price points")
    print(f"Scoring basis: R^2 vs the gold MILP (gap {GOLD_MIP_GAP:g}) OPEX, absolute [€]\n")

    df = reusable_solves(points)
    if df is None:
        df = solve_test_points(points)
        SOLVES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(solves_csv(), index=False)
        print(f"\nSaved solves to {solves_csv()} (reused by the next run)")
    return df


def features_of(mode: str, df: pd.DataFrame) -> pd.DataFrame:
    """The feature frame a mode's surrogate expects for these price points: the ratio
    c_g/c_el for the 1D family, the price pair itself for the 2D family."""
    if pipeline.TEST_KEY_OF[mode] == "1D":
        return pd.DataFrame({"ratio": df["gas_price_MWh"] / df["electricity_price_MWh"]})
    return df[["gas_price_MWh", "electricity_price_MWh"]]


def predict_absolute(mode: str, col: str, df: pd.DataFrame) -> np.ndarray:
    """A trained surrogate's prediction for these price points, in absolute €.

    2D-family surrogates predict absolute € directly. 1D-family surrogates predict *specific*
    OPEX (OPEX / c_el), so their prediction is rescaled by the real per-point electricity
    price -- the same conversion run_full_pipeline._r2_vs_milp does, and the only way a 1D
    and a 2D surrogate end up on one comparable axis.
    """
    model = joblib.load(pipeline.MODELS_DIR / mode / f"surrogate_{col}.joblib")
    pred = model.predict(features_of(mode, df))
    if pipeline.TEST_KEY_OF[mode] == "1D":
        pred = pred * (df["electricity_price_MWh"].to_numpy() / 1000.0)
    return pred


def best_formulation_of(mode: str, df: pd.DataFrame) -> tuple[str, float]:
    """Pick the surrogate to represent one family (mode) on the front: of the four LRs trained
    for it -- one per formulation's OPEX (MILP, LP lower, LP upper, LP approx) -- the one whose
    prediction best matches the gold MILP, returned as (formulation, its R^2).

    The four are all just linear maps of the same features; they differ only in the target they
    were fitted to. Scoring every one of them against the gold MILP and keeping the best asks
    the question this plot is about ("how well can this feature space predict the true OPEX?")
    rather than fixing a target a priori -- and the same rule is applied to both families, so
    their two points on the front are picked like for like.

    Only the trained models are read from Marius/surrogate_models/; the predictions are
    recomputed here on the test points this script solved.
    """
    y_true = df["opex_milp_gold"].to_numpy()
    r2_of = {col: r2_score(y_true, predict_absolute(mode, col, df)) for col in trainer.OPEX_COLUMNS}
    for col, r2 in r2_of.items():
        print(f"  {LR_MODES[mode]:4s} {col:15s} R2 = {r2:.4f}")
    best_col = max(r2_of, key=r2_of.get)
    print(f"  -> LR {LR_MODES[mode]} is represented by '{best_col}' (R2 = {r2_of[best_col]:.4f})")
    return best_col, r2_of[best_col]


def time_predict(mode: str, col: str, df: pd.DataFrame) -> float:
    """Seconds per price point for a surrogate's inference (model.predict + the 1D rescaling),
    the per-query cost. Training is excluded on purpose -- see the module docstring."""
    model = joblib.load(pipeline.MODELS_DIR / mode / f"surrogate_{col}.joblib")
    features = features_of(mode, df)
    is_1d = pipeline.TEST_KEY_OF[mode] == "1D"
    c_el = df["electricity_price_MWh"].to_numpy() / 1000.0

    start = perf_counter()
    for _ in range(PREDICT_REPEATS):
        pred = model.predict(features)
        if is_1d:
            pred = pred * c_el  # specific OPEX -> € , part of the per-query cost
    elapsed = perf_counter() - start

    return elapsed / PREDICT_REPEATS / len(df)


def plot_pareto(rows: list[dict], out_stem: Path):
    """Scatter the methods in the (compute time, error) plane -- both axes are quantities to
    MINIMIZE, so this reads like a conventional Pareto plot: down-and-left is better, and the
    front is the lower-left staircase.

    Accuracy is plotted as the error 1 - R^2 rather than R^2 itself. Time is log-scaled since
    the methods are orders of magnitude apart.
    """
    apply_style(width_cm=16, aspect="golden", grid=True, strict=True)
    fig, ax = plt.subplots(constrained_layout=True)

    # Fainter than the style's default grid: here it is only a reading aid behind the markers
    # and their labels, which are the content.
    ax.grid(alpha=0.25, linewidth=0.4)

    for row in rows:
        ax.scatter(row["seconds_per_point"], row["error"],
                   s=90, color=COLORS[row["method"]], edgecolors="black",
                   zorder=3, label=row["method"])
        # The two MILPs sit close together in time, so their labels would overlap if both
        # were drawn above the marker -- the loose one carries label_dy to move below it.
        ax.annotate(row["label"],
                    (row["seconds_per_point"], row["error"]),
                    textcoords="offset points", xytext=(0, row.get("label_dy", 12)),
                    ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=4)

    # The Pareto front, now that BOTH axes are minimized: walk left to right in time and keep
    # a point only if nothing cheaper already achieved an error this low.
    front, best_so_far = [], np.inf
    for row in sorted(rows, key=lambda r: r["seconds_per_point"]):
        if row["error"] < best_so_far:
            front.append(row)
            best_so_far = row["error"]
    ax.plot([r["seconds_per_point"] for r in front], [r["error"] for r in front],
            "-", color=FRONT_COLOR, linewidth=1.6, zorder=2, label="Pareto front")

    ax.set_xscale("log")

    # Pad both axes so the point labels (which sit above the markers) are not clipped by the
    # axes box -- the points themselves span the full data range.
    times = [row["seconds_per_point"] for row in rows]
    errors = [row["error"] for row in rows]
    decades = np.log10(max(times)) - np.log10(min(times))
    ax.set_xlim(min(times) * 10 ** (-0.25 * decades), max(times) * 10 ** (0.25 * decades))
    span = max(errors) - min(errors)
    ax.set_ylim(min(errors) - 0.10 * span, max(errors) + 0.18 * span)

    ax.set_xlabel("Mean compute time per test sample [s]")
    ax.set_ylabel(f"$1 - R^2$ vs. {GOLD_METHOD} OPEX [€]")
    # The mathtext labels ($LP^L$, ...) are taller than the plain ones, which would make the
    # legend rows unevenly spaced. handleheight fixes a row height that exceeds every label's,
    # so the rows come out evenly spaced regardless of what is in them.
    ax.legend(fontsize=8, loc="upper right", handleheight=1.6, labelspacing=0.5)

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    fig.savefig(out_png)  # dpi/bbox come from apply_style's rcParams
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\nSaved Pareto plot to {out_png} and {out_pdf}")


def main():
    pin_pipeline_test_set()
    df = test_solves()

    y_gold = df["opex_milp_gold"]
    r2_loose = r2_score(y_gold, df["opex_milp"])
    r2_lp = r2_score(y_gold, (df["opex_lp_lower"] + df["opex_lp_upper"]) / 2.0)
    t_gold = df["time_milp_gold"].mean()

    rows = [
        {"method": GOLD_METHOD, "label": GOLD_METHOD,
         "seconds_per_point": t_gold, "r2_vs_gold": 1.0},
        {"method": LOOSE_METHOD, "label": LOOSE_METHOD, "label_dy": -13,
         "seconds_per_point": df["time_milp"].mean(), "r2_vs_gold": r2_loose},
        # The legend already spells the mean out, so the marker annotation stays short.
        {"method": LP_MEAN_METHOD, "label": "LP mean",
         # The LP mean needs BOTH bounds, so its cost is the lower solve plus the upper solve.
         "seconds_per_point": (df["time_lp_lower"] + df["time_lp_upper"]).mean(),
         "r2_vs_gold": r2_lp},
    ]

    # One point per surrogate family, each represented by its own best-fitting formulation.
    print("\nR^2 vs the gold MILP (absolute [€]) for each family's four trained surrogates:")
    lr_row_of = {}
    for mode, name in LR_MODES.items():
        col, r2_lr = best_formulation_of(mode, df)
        lr_row_of[mode] = {
            "method": f"Best {name} LR",  # legend entry
            "label": f"LR {name}: {FORMULATION_LABELS[col]}",  # annotation next to the marker
            # Both LRs cost the same one dot product, so they sit on top of each other in
            # time; the 2D label goes below its marker so the two cannot overlap.
            "label_dy": -13 if name == "2D" else 11,
            "seconds_per_point": time_predict(mode, col, df),
            "r2_vs_gold": r2_lr,
        }
    rows.extend(lr_row_of.values())

    # Both plotted axes are minimized, so accuracy enters as the error 1 - R^2.
    for row in rows:
        row["error"] = 1.0 - row["r2_vs_gold"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"pareto_accuracy_vs_time_{pipeline.test_tag()}.csv"
    table = pd.DataFrame(rows)[["method", "label", "seconds_per_point", "r2_vs_gold", "error"]]
    table.to_csv(out_csv, index=False)
    print(f"\nSaved Pareto table to {out_csv}")
    for row in rows:
        print(f"  {row['method']:18s} {row['seconds_per_point']:12.6f} s/point   "
              f"R2 = {row['r2_vs_gold']:.4f}   1-R2 = {row['error']:.4f}")

    # Each LR's excluded up-front cost, for reference: its training set had to be solved once.
    # Those solves were timed by the pipeline, so report what they actually cost -- the only
    # thing here that cannot be measured in this run, since this script does not train.
    print("\nFor reference, each LR's one-off training cost (deliberately NOT on the "
          "per-query axis):")
    for mode, name in LR_MODES.items():
        train_csv = ROOT / (f"Marius/results/evaluation_{pipeline.N_TRAIN}_training_samples_"
                            f"{pipeline.TRAIN_KEY_OF[mode]}.csv")
        if not train_csv.exists():
            print(f"  LR {name}: no training set on disk at {train_csv}")
            continue
        df_train = pd.read_csv(train_csv)
        if "time_milp" in df_train.columns:
            train_seconds = df_train["time_milp"].sum()
            payback = train_seconds / (t_gold - lr_row_of[mode]["seconds_per_point"])
            print(f"  LR {name}: {len(df_train)} MILP solves = {train_seconds:.1f} s, i.e. it "
                  f"pays for itself after {payback:,.0f} queries versus solving the gold MILP "
                  f"each time.")

    plot_pareto(rows, OUT_DIR / f"pareto_accuracy_vs_time_{pipeline.test_tag()}")


if __name__ == "__main__":
    main()
