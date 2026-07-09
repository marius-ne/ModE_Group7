from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

# Existing project artefacts. Replace these paths if you want to compare other models.
TEST_SAMPLE_CSV_CANDIDATES = [
    ROOT / "Marius" / "results" / "evaluation_test_samples_2D.csv",
    ROOT / "Marius" / "results" / "evaluation_10_test_samples_2D.csv",
]
ABSOLUTE_3D_MODEL_PATH = (
    ROOT / "Florian" / "surrogate_models" / "joblibs" / "surrogate_model_2d_prices_40_opex_milp.joblib"
)
RATIO_MODEL_PATH = ROOT / "Florian" / "validation" / "joblibs" / "40_ratio_opex_milp.joblib"

TARGET_COLUMN = "opex_milp"
OUTPUT_PNG = ROOT / "ratio_vs_absolute_regression_error_bubbles.png"


@dataclass
class MockModel:
    feature_names_in_: tuple[str, ...]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if tuple(self.feature_names_in_) == ("ratio",):
            return 60000 + 90000 * x["ratio"].to_numpy()
        return 20000 + 80000 * x["c_G"].to_numpy() + 65000 * x["c_el"].to_numpy()


def load_model_or_mock(path: Path, feature_names: tuple[str, ...]):
    if path.exists():
        return joblib.load(path)

    print(f"Model not found, using mock model instead: {path}")
    return MockModel(feature_names_in_=feature_names)


def load_test_data() -> pd.DataFrame:
    for path in TEST_SAMPLE_CSV_CANDIDATES:
        if path.exists():
            print(f"Using test sample: {path}")
            return pd.read_csv(path)

    print(f"No test sample found, using mock test data instead. Checked: {TEST_SAMPLE_CSV_CANDIDATES}")
    return pd.DataFrame(
        {
            "gas_price_MWh": [65, 130, 235, 266, 90, 180, 310, 120],
            "electricity_price_MWh": [260, 406, 385, 334, 180, 450, 510, 230],
            TARGET_COLUMN: [18500, 33100, 47400, 49700, 21000, 44000, 69000, 28000],
        }
    )


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"None of these columns exist in the test data: {candidates}")


def to_eur_per_kwh(values: pd.Series, column_name: str) -> pd.Series:
    if column_name.endswith("_MWh"):
        return values / 1000.0
    return values


def prepare_inputs(df: pd.DataFrame) -> pd.DataFrame:
    gas_col = first_existing_column(df, ["c_G", "gas_price", "gas_price_MWh"])
    electricity_col = first_existing_column(
        df,
        ["c_el", "c_e", "actual_c_electricity", "electricity_price", "electricity_price_MWh"],
    )

    prepared = pd.DataFrame(index=df.index)
    prepared["c_G_raw"] = df[gas_col]
    prepared["c_el_raw"] = df[electricity_col]
    prepared["c_G"] = to_eur_per_kwh(df[gas_col], gas_col)
    prepared["c_el"] = to_eur_per_kwh(df[electricity_col], electricity_col)
    prepared["ratio"] = prepared["c_G"] / prepared["c_el"]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in the test data.")
    prepared["opex_true"] = df[TARGET_COLUMN]
    return prepared


def predict_absolute_opex(prepared: pd.DataFrame) -> pd.DataFrame:
    absolute_model = load_model_or_mock(ABSOLUTE_3D_MODEL_PATH, ("c_G", "c_el"))
    ratio_model = load_model_or_mock(RATIO_MODEL_PATH, ("ratio",))

    absolute_input = pd.DataFrame({"c_G": prepared["c_G"], "c_el": prepared["c_el"]})
    ratio_input = pd.DataFrame({"ratio": prepared["ratio"]})

    result = prepared.copy()
    result["opex_pred_3d"] = absolute_model.predict(absolute_input)
    result["opex_pred_ratio_specific"] = ratio_model.predict(ratio_input)
    result["opex_pred_ratio_absolute"] = result["opex_pred_ratio_specific"] * result["c_el"]
    result["error_3d"] = (result["opex_true"] - result["opex_pred_3d"]).abs()
    result["error_2d"] = (result["opex_true"] - result["opex_pred_ratio_absolute"]).abs()
    return result


def bubble_sizes(errors: pd.Series) -> np.ndarray:
    min_size = 60
    max_size = 900
    if np.isclose(errors.max(), errors.min()):
        return np.full(len(errors), (min_size + max_size) / 2)
    scaled = (errors - errors.min()) / (errors.max() - errors.min())
    return min_size + scaled.to_numpy() * (max_size - min_size)


def add_ratio_weight_background(ax, x_values: pd.Series, y_values: pd.Series) -> None:
    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()
    x_grid = np.linspace(x_min, x_max, 300)
    penalty = 1.0 / np.maximum(x_grid, 1e-12) ** 2
    penalty = (penalty - penalty.min()) / (penalty.max() - penalty.min())
    background = np.tile(penalty, (100, 1))

    ax.imshow(
        background,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="auto",
        cmap="Reds",
        alpha=0.35,
        zorder=0,
    )


def add_constant_ratio_lines(ax, x_values: pd.Series, y_values: pd.Series) -> None:
    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()
    x_line = np.linspace(x_min, x_max, 200)
    ratios = np.quantile(y_values / x_values, [0.25, 0.5, 0.75])

    for ratio in ratios:
        y_line = ratio * x_line
        mask = (y_line >= y_min) & (y_line <= y_max)
        if not mask.any():
            continue
        ax.plot(
            x_line[mask],
            y_line[mask],
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label=f"C_gas / C_el = {ratio:.2f}",
            zorder=2,
        )


def plot_error_comparison(result: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True, sharey=True)
    color_min = min(result["error_3d"].min(), result["error_2d"].min())
    color_max = max(result["error_3d"].max(), result["error_2d"].max())

    plot_specs = [
        (axes[0], "3D absolute regression", "error_3d", "tab:blue"),
        (axes[1], "2D ratio regression converted to absolute OPEX", "error_2d", "tab:red"),
    ]

    x = result["c_el_raw"]
    y = result["c_G_raw"]

    for ax, title, error_col, color in plot_specs:
        add_ratio_weight_background(ax, x, y)
        add_constant_ratio_lines(ax, x, y)
        sizes = bubble_sizes(result[error_col])
        scatter = ax.scatter(
            x,
            y,
            s=sizes,
            c=result[error_col],
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.85,
            zorder=3,
        )
        ax.set_title(title)
        ax.set_xlabel("Electricity price C_el [EUR/MWh]")
        ax.grid(alpha=0.25, zorder=1)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
        cbar.set_label("Absolute OPEX error [EUR]")

    axes[0].set_ylabel("Gas price C_gas [EUR/MWh]")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=min(3, len(unique)),
        title="Constant ratio lines",
        bbox_to_anchor=(0.5, -0.02),
    )

    size_handles = [
        plt.scatter([], [], s=size, color="lightgray", edgecolors="black")
        for size in [100, 400, 800]
    ]
    axes[1].legend(
        size_handles,
        ["small error", "medium error", "large error"],
        title="Bubble size",
        loc="upper right",
        framealpha=0.9,
    )

    fig.suptitle(
        "Absolute OPEX error over the gas/electricity price plane\n"
        "Red background shows the implicit ratio-model penalty 1 / C_el^2",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PNG


def main() -> None:
    test_df = load_test_data()
    prepared = prepare_inputs(test_df)
    result = predict_absolute_opex(prepared)
    output_path = plot_error_comparison(result)

    print("Preview of calculated errors:")
    print(result[["c_el_raw", "c_G_raw", "opex_true", "opex_pred_3d", "opex_pred_ratio_absolute", "error_3d", "error_2d"]])
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
