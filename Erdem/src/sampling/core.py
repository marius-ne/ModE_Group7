#src/sampling/run_sampling.py
"""
Sampling module for creating the training and test dataset for the surrogate model
with latin hypercube sampling (LHS) and sobol sequencing.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import qmc
from src.misc.constants import RESULTS_DIR

# == Sampling parameters =========================

# -- Price ranges --------------------------
GAS_MIN, GAS_MAX = 25.0, 315.0 # €/MWh
ELEC_MIN, ELEC_MAX = 65.0, 450.0 # €/MWh

# -- Number of sample points ----------------------
N_TOTAL = 40 # total number of sample points (including corners and edge midpoints)
N_INTERIOR = 32 # interior points per method
                # Sobol: must be power of 2 → maximum 32 points with 40 points available
                # LHS: no restriction → 32 chosen to keep sets comparable
N_CORNERS = 4 # corners of the price space [GAS_MIN, GAS_MAX] x [ELEC_MIN, ELEC_MAX] to prevent extrapolation in the surrogate model
N_EDGES = 4

MARGIN = 0.025   # 2.5 % inset in [0,1]-space so interior pts stay clear of boundary

# -- Boundary points ---------------------
corners = np.array([
    [GAS_MIN, ELEC_MIN], # bottom-left
    [GAS_MAX, ELEC_MIN], # bottom-right
    [GAS_MIN, ELEC_MAX], # top-left
    [GAS_MAX, ELEC_MAX], # top-right
])

gas_mid  = (GAS_MIN + GAS_MAX) / 2
elec_mid = (ELEC_MIN + ELEC_MAX) / 2

edge_midpoints = np.array([
    [gas_mid, ELEC_MIN], # bottom edge center
    [gas_mid, ELEC_MAX], # top edge center
    [GAS_MIN, elec_mid], # left edge center
    [GAS_MAX, elec_mid], # right edge center
])

boundary_pts = np.vstack([corners, edge_midpoints])


# == Helper functions =========================
def scale_to_price_domain(sample):
    """
    Scale created sample from [0,1]^2-space to the actual price domain
    :param sample: Array in [0,1]^2
    :return: Scaled array with margin in [GAS_MIN, GAS_MAX] x [ELEC_MIN, ELEC_MAX]
    """
    inset = MARGIN + sample * (1 - 2 * MARGIN) # scale to [MARGIN, 1-MARGIN]^2 to keep interior points away from boundaries
    return qmc.scale(
        inset,
        l_bounds=[GAS_MIN, ELEC_MIN],
        u_bounds=[GAS_MAX, ELEC_MAX]
    )


def create_df(
        interior_pts,
        n_samples: int = N_TOTAL,
        n_interior: int = N_INTERIOR,
        n_corner: int = N_CORNERS,
):
    """
    Create a DataFrame with the sampled points and their corresponding labels (corner, edge_midpoint, interior)
    :param interior_pts: Scaled array with margin in [GAS_MIN, GAS_MAX] x [ELEC_MIN, ELEC_MAX]
    :param n_samples: Number of total sample points to create
    :param n_interior: Number of samples points in the interior of the price domain
    :param n_corner: Number of sample points on corners of the price domain
    :return: DataFrame with columns "gas_price", "electricity_price", "point_type" and the corresponding points as a numpy array
    """
    labels = (["corner"] * n_corner +
              ["edge_midpoint"] * (n_samples - n_corner - n_interior) +
              ["interior"] * n_interior)
    pts = np.vstack([boundary_pts, interior_pts])
    return pd.DataFrame({
        "gas_price": pts[:, 0],
        "electricity_price": pts[:, 1],
        "point_type": labels,
    }), pts


def quality(pts):
    """
    Measures the quality of the sampling method by calculating the discrepancy and minimum pairwise distance
    the sampling points.
    :param pts: Numpy array containing the sampling points in the price domain
    :return: Dictionary of discrepancy and minimum pairwise distance
    """
    unit = qmc.scale(pts,
                     l_bounds=[GAS_MIN,  ELEC_MIN],
                     u_bounds=[GAS_MAX,  ELEC_MAX],
                     reverse=True)
    disc = qmc.discrepancy(unit)
    n = len(pts)
    dists = [np.linalg.norm(pts[i] - pts[j])
             for i in range(n) for j in range(i + 1, n)]
    sample_quality = {
        "discrepancy": disc,
        "min_distance": min(dists)
    }

    return sample_quality

def print_summary(name, disc, min_dist):
    """
    Print a summary of the sampling method's quality metrics (discrepancy and minimum pairwise distance)
    :param name: Sampling method's name
    :param disc: Discrepancy of the sampling points, which measures how uniformly the points are distributed in the sample space.
    Lower values indicate better uniformity.
    :param min_dist: Minimum pairwise distance of the sampling points.
    """
    print(f"{name:<6} \n- discrepancy = {disc:.5f}\n- min pairwise dist = {min_dist:.2f} €/MWh")


def save_samples(
        df_samples: pd.DataFrame,
        sample_quality: dict,
        sampling_method: str,
        file_name: str
):
    """
    Save the sampled points and their quality metrics to a parquet file
    :param df_samples: DataFrame with columns "gas_price", "electricity_price", "point_type" and the corresponding values
    :param sample_quality: Dictionary of discrepancy and minimum pairwise distance
    :param sampling_method: Name of the sampling method (either "lhs" or "sobol")
    :param file_name: Name of the output file (without extension) to save the samples (either "training" or "test")
    """
    base_dir = RESULTS_DIR / "Sampling"
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        sample_file = base_dir / f"{sampling_method}_{file_name}_samples.parquet"
        df_samples.to_parquet(sample_file)

        print(f"Saved {sampling_method} {file_name} samples to {sample_file}")

        quality_file = base_dir / f"{sampling_method}_{file_name}_quality.json"
        with open(quality_file, "w") as f:
            json.dump(sample_quality, f)
        print(f"Saved {sampling_method} {file_name} sample quality metrics to {quality_file}")

    except Exception as e:
        raise e


def load_samples(sampling_method: str, file_name: str) -> tuple[pd.DataFrame, dict]:
    """
    Load the sampled points and their quality metrics.
    :param sampling_method: Name of the sampling method (either "lhs" or "sobol")
    :param file_name: Name of the output file (without extension) to save the samples (either "training" or "test")
    :return: DataFrame with the sample points and a dictionary with the sample quality metrics
    """
    base_dir = RESULTS_DIR / "Sampling"

    sample_file = base_dir / f"{sampling_method}_{file_name}_samples.parquet"
    df_samples = pd.read_parquet(sample_file)

    quality_file = base_dir / f"{sampling_method}_{file_name}_quality.json"
    with open(quality_file, "r") as f:
        dict_quality = json.load(f)

    return df_samples, dict_quality


# == Sampling function ===========================
def create_sample(
        sampling_method: str,
        n_samples: int = N_TOTAL,
        n_interior: int = N_INTERIOR,
        n_corner: int = N_CORNERS,
) -> tuple[pd.DataFrame, dict]:
    """
    Create a sample set using the specified sampling method and return a DataFrame with the sampling points and a dictionary with quality metrics
    :param sampling_method: The sampling method to use ("sobol" or "lhs")
    :param n_samples: Number of total sample points to create
    :param n_interior: Number of samples points in the interior of the price domain
    :param n_corner: Number of sample points on corners of the price domain
    :return: DataFrame with the sampling points and dictionary with quality metrics (discrepancy and minimum pairwise distance) of the sampling method
    """
    if sampling_method == "sobol":
        sampler = qmc.Sobol(d=2, scramble=True, seed=28)

    elif sampling_method == "lhs":
        sampler = qmc.LatinHypercube(d=2, scramble=True, optimization="random-cd", seed=28)

    else:
        raise f"Sampling method {sampling_method} not supported. Choose either 'sobol' or 'lhs'."

    raw_samples = sampler.random(n_interior) # [0,1]^2
    sample_interior = scale_to_price_domain(raw_samples)

    df_samples, arr_samples = create_df(sample_interior, n_samples, n_interior, n_corner)
    sample_quality = quality(arr_samples)

    return df_samples, sample_quality


# == Visualization functions ======================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# == Visualization constants ======================
COLORS = {"corner": "#D85A30", "edge_midpoint": "#BA7517", "interior": "#1D9E75"}
MARKERS = {"corner": "D", "edge_midpoint": "s", "interior": "o"}
SIZES = {"corner": 90, "edge_midpoint": 65, "interior": 35}


def plot_samples(
        df_sobol: pd.DataFrame,
        sobol_quality: dict,
        df_lhs: pd.DataFrame,
        lhs_quality: dict,
        plot_file: str = "sampling_comparison.png",
):
    """
    Create a side-by-side comparison plot of Sobol and LHS sampling methods.
    :param df_sobol: DataFrame with Sobol sample points
    :param sobol_quality: Dictionary with "discrepancy" and "min_distance" for Sobol samples
    :param df_lhs: DataFrame with LHS sample points
    :param lhs_quality: Dictionary with "discrepancy" and "min_distance" for LHS samples
    :param plot_file: Path/filename for saving the plot (default: "sampling_comparison.png")
    """

    def _draw_boundary(ax):
        """Draw the price domain boundary rectangle."""
        rect = mpatches.FancyBboxPatch(
            (GAS_MIN, ELEC_MIN),
            GAS_MAX - GAS_MIN, ELEC_MAX - ELEC_MIN,
            boxstyle="square,pad=0", linewidth=1,
            edgecolor="#888780", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

    def _scatter_panel(ax, df, title, disc, min_dist):
        """Draw a single scatter plot panel with quality metrics."""
        _draw_boundary(ax)

        for ptype in ["interior", "edge_midpoint", "corner"]:
            mask = df["point_type"] == ptype
            ax.scatter(
                df.loc[mask, "gas_price"],
                df.loc[mask, "electricity_price"],
                c=COLORS[ptype], marker=MARKERS[ptype], s=SIZES[ptype],
                edgecolors="white", linewidths=0.5,
                zorder={"corner": 3, "edge_midpoint": 2, "interior": 1}[ptype],
            )

        ax.set_xlim(GAS_MIN - 7, GAS_MAX + 7)
        ax.set_ylim(ELEC_MIN - 12, ELEC_MAX + 12)
        ax.set_xlabel("Gas price (€/MWh)", fontsize=10)
        ax.set_ylabel("Electricity price (€/MWh)", fontsize=10)
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(True, linewidth=0.4, alpha=0.5)

        # Metric annotation box
        info = (f"Discrepancy:  {disc:.5f}\n"
                f"Min dist:       {min_dist:.1f} €/MWh")
        ax.text(0.03, 0.97, info,
                transform=ax.transAxes, fontsize=8.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#cccccc", alpha=0.55))

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    # Plot Sobol and LHS samples
    _scatter_panel(
        axes[0], df_sobol,
        f"Sobol sequence  ({N_INTERIOR} interior + {N_CORNERS + N_EDGES} boundary = {N_TOTAL})",
        sobol_quality["discrepancy"], sobol_quality["min_distance"]
    )
    _scatter_panel(
        axes[1], df_lhs,
        f"Latin hypercube  ({N_INTERIOR} interior + {N_CORNERS + N_EDGES} boundary = {N_TOTAL})",
        lhs_quality["discrepancy"], lhs_quality["min_distance"]
    )

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=COLORS["corner"], label=f"Corner ({N_CORNERS})"),
        mpatches.Patch(color=COLORS["edge_midpoint"], label=f"Edge midpoint ({N_EDGES})"),
        mpatches.Patch(color=COLORS["interior"], label=f"Interior ({N_INTERIOR})"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04)
    )

    # Title
    plt.suptitle(
        f"Sampling comparison — {N_TOTAL} points | "
        f"gas {GAS_MIN}–{GAS_MAX}€/MWh,  elec {ELEC_MIN}–{ELEC_MAX}€/MWh",
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    plot_file = RESULTS_DIR / "Sampling" / plot_file
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to {plot_file}")