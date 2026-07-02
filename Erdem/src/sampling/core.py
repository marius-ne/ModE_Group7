#src/sampling/run_sampling.py
"""
Sampling module for creating the training and test dataset for the surrogate model
with latin hypercube sampling (LHS) and sobol sequencing.
"""
import numpy as np
import pandas as pd
import json

from scipy.stats import qmc
from src.misc.constants import RESULTS_DIR

# == Sampling parameters =========================

# -- Price ranges --------------------------
GAS_MIN, GAS_MAX = 25.0, 315.0 # €/MWh
ELEC_MIN, ELEC_MAX = 65.0, 450.0 # €/MWh

MARGIN = 0.0   # XX % inset in [0,1]-space so interior pts stay clear of boundary

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


def _largest_power_of_two_leq(n: int) -> int:
    """
    Return the largest power of two <= n (at least 1).
    :param n: Number of samples
    :return: Largest power of two
    """
    if n < 1:
        return 1
    m = int(np.floor(np.log2(n)))
    return 2 ** m


def create_df(
        interior_pts: np.ndarray,
        n_total: int,
        n_corner: int = 4,
        n_edges: int = 4,
        include_bounds: bool = True
):
    """
    Create a DataFrame with the sampled points and their corresponding labels (corner, edge_midpoint, interior)
    :param interior_pts: Scaled array with margin in [GAS_MIN, GAS_MAX] x [ELEC_MIN, ELEC_MAX]
    :param n_total: Number of total sample points to create
    :param n_corner: Number of sample points on corners of the price domain
    :param n_edges: Number of sample points on edges of the price domain
    :param include_bounds: Whether to include the boundary points (corners and edge midpoints) in the DataFrame
    :return: DataFrame with columns "gas_price", "electricity_price", "point_type" and the corresponding points as a numpy array
    """
    interior_pts = np.asarray(interior_pts)
    if include_bounds:
        n_interior = len(interior_pts)
        remain = n_total - n_corner - n_interior

        if remain < 0:
            remain = 0

        n_edges_used = min(n_edges, max(0, remain))
        boundary_to_use = np.vstack([corners[:n_corner], edge_midpoints[:n_edges_used]]) if (n_corner + n_edges_used) > 0 else np.empty((0, 2))
        pts = np.vstack([boundary_to_use, interior_pts])
        labels = (["corner"] * n_corner) + (["edge_midpoint"] * n_edges_used) + (["interior"] * n_interior)

    else:
        pts = interior_pts
        labels = ["interior"] * len(interior_pts)

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
        samples: pd.DataFrame | pd.Series,
        sample_quality: dict | None,
        sampling_method: str,
        sample_type: str,
        file_name: str | int | None = None
):
    """
    Save the sampled points and their quality metrics to a parquet and JSON file or a csv file (for log)
    :param samples: DataFrame or Series with columns "gas_price", "electricity_price", "point_type" and the corresponding values
    :param sample_quality: Dictionary of discrepancy and minimum pairwise distance
    :param sampling_method: Name of the sampling method (either "lhs", "sobol", "log or "random")
    :param file_name: Optional suffix of the output file (without extension)
    :param sample_type: Type of sampling (either "training" or "test")
    """
    base_dir = RESULTS_DIR / "Sampling" / sample_type
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        if sampling_method == "log":
            # Save samples as csv for log (1D)
            suffix = f"_{file_name}" if file_name else ""
            sample_file = base_dir / f"{sampling_method}{suffix}_samples.csv"
            samples.to_csv(sample_file, index=False, header=True, float_format="%.18g")
            print(f"Saved {sampling_method} {file_name} samples to {sample_file}")

        else:
            # Save samples as parquet for Sobol, LHS and random (2D)
            suffix = f"_{file_name}" if file_name else ""
            sample_file = base_dir / f"{sampling_method}{suffix}_samples.parquet"
            samples.to_parquet(sample_file)
            print(f"Saved {sampling_method} {file_name} samples to {sample_file}")

            # Save quality file as json
            if sample_quality is not None:
                quality_file = base_dir / f"{sampling_method}{suffix}_quality.json"
                with open(quality_file, "w") as f:
                    json.dump(sample_quality, f)
                print(f"Saved {sampling_method} {file_name} sample quality metrics to {quality_file}")

    except Exception as e:
        raise e


def load_samples(
        sampling_method: str,
        sample_type: str,
        file_name: str | None = None,
) -> tuple[pd.DataFrame | pd.Series, dict | None]:
    """
    Load the sampled points and their quality metrics.
    :param sampling_method: Name of the sampling method (either "lhs", "sobol", "log" or "random")
    :param sample_type: Type of sampling (either "training" or "test")
    :param file_name: Optional suffix of the output file (without extension)
    :return: DataFrame with the sample points and a dictionary with the sample quality metrics
    or just Series with the sample points
    """
    base_dir = RESULTS_DIR / "Sampling" / sample_type

    if sampling_method == "log":
        suffix = f"_{file_name}" if file_name else ""
        sample_file = base_dir / f"{sampling_method}{suffix}_samples.csv"

        samples = pd.read_csv(sample_file)
        dict_quality = None

        return samples, dict_quality

    else:
        suffix = f"_{file_name}" if file_name else ""
        sample_file = base_dir / f"{sampling_method}{suffix}_samples.parquet"
        samples = pd.read_parquet(sample_file)

        # Only load quality metrics for sobol/lhs
        quality_file = base_dir / f"{sampling_method}{suffix}_quality.json"
        with open(quality_file, "r") as f:
            dict_quality = json.load(f)

        return samples, dict_quality


# == Sampling function ===========================
def create_sample(
        sampling_method: str,
        n_total: int,
        n_corner: int = 4,
        n_edges: int = 4
) -> tuple[pd.DataFrame, dict] | pd.Series:
    """
    Create a sample set using the specified sampling method and return a DataFrame with the sampling points and a dictionary with quality metrics
    :param sampling_method: The sampling method to use ("sobol", "lhs", "log" or "random")
    :param n_total: Number of total sample points to create
    :param n_corner: Number of sample points on corners of the price domain
    :param n_edges: Number of sample points on edges of the price domain
    :return: DataFrame with the sampling points and dictionary with quality metrics (discrepancy and minimum pairwise distance) of the sampling method
    or just a Series with the sampling points
    """
    method = sampling_method.lower()

    if method == "log":
        # Generate logarithmically spaced samples of the ratio of gas to electricity prices (1D)
        gas_prices = np.array([GAS_MIN, GAS_MAX])
        electr_prices = np.array([ELEC_MIN, ELEC_MAX])

        # Define all ratio combinations and filter min and max ratio
        ratio_matrix = gas_prices[:, None] / electr_prices[None, :]
        min_ratio = ratio_matrix.min()
        max_ratio = ratio_matrix.max()

        # Create evenly spaced sampling points on a log-scale
        samples = np.logspace(np.log10(min_ratio), np.log10(max_ratio), n_total)

        return pd.Series(samples, name="ratios")

    elif method == "random":
        # Generate random samples within the price domain (2D)
        raw_samples = np.random.uniform(0, 1, size=(n_total, 2))
        samples = scale_to_price_domain(raw_samples)

        df_samples, arr_samples = create_df(
            samples,
            n_total,
            n_corner = 0,
            n_edges = 0,
            include_bounds = False
        )
        sample_quality = quality(arr_samples)

        return df_samples, sample_quality

    if method == "lhs":
        # reserve corners: n_interior = n_total - n_corner
        n_interior = max(0, n_total - n_corner)

        # LH sampling for interior
        sampler = qmc.LatinHypercube(d=2, scramble=True, optimization="random-cd", seed=28) # generate n_interior LHS points in [0,1]^2
        raw_samples = sampler.random(n_interior) if n_interior > 0 else np.empty((0,2))
        sample_interior = scale_to_price_domain(raw_samples) if n_interior > 0 else np.empty((0,2))

        # include_bounds True: include corners but not edge midpoints
        df_samples, arr_samples = create_df(sample_interior, n_total=n_total, n_corner=n_corner, n_edges=0, include_bounds=True)
        sample_quality = quality(arr_samples)

        return df_samples, sample_quality

    if method == "sobol":

        # available slots after reserving corners
        available = max(0, n_total - n_corner)

        # Sobol interior: choose largest power of two <= available
        n_interior_sobol = _largest_power_of_two_leq(available)

        # remaining slots after sobol interior
        remain = available - n_interior_sobol

        # allocate edge midpoints up to n_edges
        n_edges_used = min(n_edges, remain)
        remain_after_edges = remain - n_edges_used

        # any leftover points become additional random interior points
        n_extra_random_interior = remain_after_edges

        # sample sobol interior (exact 2^m samples)
        if n_interior_sobol > 0:
            m = int(np.log2(n_interior_sobol))
            sampler = qmc.Sobol(d=2, scramble=True, seed=28)

            try:
                raw_sobol = sampler.random_base2(m)

            except Exception:
                raw_sobol = sampler.random(n_interior_sobol)

            sobol_interior = scale_to_price_domain(raw_sobol)

        else:
            sobol_interior = np.empty((0,2))

        # additional random interior points if needed
        if n_extra_random_interior > 0:
            raw_rand = np.random.uniform(0.0, 1.0, size=(n_extra_random_interior, 2))
            rand_interior = scale_to_price_domain(raw_rand)

        else:
            rand_interior = np.empty((0,2))

        # combine interior points: sobol first, then random extras
        interior_all = np.vstack([sobol_interior, rand_interior]) if (len(sobol_interior) + len(rand_interior)) > 0 else np.empty((0,2))

        # Now build final DataFrame: include corners and n_edges_used edge midpoints
        df_samples, arr_samples = create_df(interior_all, n_total=n_total, n_corner=n_corner, n_edges=n_edges_used, include_bounds=True)
        sample_quality = quality(arr_samples)

        return df_samples, sample_quality

    else:
        raise ValueError(
            f"Sampling method '{sampling_method}' not supported. "
            "Choose either 'sobol', 'lhs', 'log' or 'random'."
        )


def generate_test_sample(n_samples: int):
    """
    Create a random test sample using Latin hypercube sampling within the price boundaries.
    :param n_samples: Number of test samples to create
    :return: DataFrame with column "ratios"
    """
    sampler = qmc.LatinHypercube(d=2, scramble=True, optimization="random-cd", seed=28)
    raw_samples = sampler.random(n_samples)
    test_samples = qmc.scale(
        raw_samples,
        l_bounds=[GAS_MIN, ELEC_MIN],
        u_bounds=[GAS_MAX, ELEC_MAX]
    )
    ratios = test_samples[:, 0] / test_samples[:, 1]

    df_test_samples = pd.DataFrame({
        "ratios": ratios,
    })

    base_dir = RESULTS_DIR / "Sampling"
    base_dir.mkdir(parents=True, exist_ok=True)
    sample_file = base_dir / f"random_sample_{n_samples}.csv"
    df_test_samples.to_csv(sample_file, index=False, header=True, float_format="%.18g")
    print(f"Saved random sample to {sample_file}")

    return df_test_samples


# == Visualization functions ======================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    def _counts_from_df(df):
        n_interior = int((df["point_type"] == "interior").sum())
        n_corner = int((df["point_type"] == "corner").sum())
        n_edges = int((df["point_type"] == "edge_midpoint").sum())
        n_total = len(df)
        return n_interior, n_corner, n_edges, n_total

    sobol_interior, sobol_corner, sobol_edges, sobol_total = _counts_from_df(df_sobol)
    lhs_interior, lhs_corner, lhs_edges, lhs_total = _counts_from_df(df_lhs)

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
        f"Sobol sequence  ({sobol_interior} interior + {sobol_corner + sobol_edges} boundary = {sobol_total})",
        sobol_quality["discrepancy"], sobol_quality["min_distance"]
    )
    _scatter_panel(
        axes[1], df_lhs,
        f"Latin hypercube  ({lhs_interior} interior + {lhs_corner + lhs_edges} boundary = {lhs_total})",
        lhs_quality["discrepancy"], lhs_quality["min_distance"]
    )

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=COLORS["corner"], label=f"Corner ({sobol_corner})"),
        mpatches.Patch(color=COLORS["edge_midpoint"], label=f"Edge midpoint ({sobol_edges})"),
        mpatches.Patch(color=COLORS["interior"], label=f"Interior ({sobol_interior})"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04)
    )

    # Title
    plt.suptitle(
        f"Sampling comparison — {sobol_total} points | "
        f"gas {GAS_MIN}–{GAS_MAX}€/MWh,  elec {ELEC_MIN}–{ELEC_MAX}€/MWh",
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    plot_file = RESULTS_DIR / "Sampling" / plot_file
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to {plot_file}")
