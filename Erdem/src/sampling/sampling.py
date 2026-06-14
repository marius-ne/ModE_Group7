#src/sampling/sampling.py
"""
Sampling module for creating the training and test dataset for the surrogate model
with latin hypercube sampling (LHS) and sobol sequencing.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

# == Sampling parameters =========================

# -- Price ranges --------------------------
GAS_MIN, GAS_MAX = 25.0, 315.0 # €/MWh
ELEC_MIN, ELEC_MAX = 65.0, 450.0 # €/MWh

# -- Number of sample points ----------------------
N_INTERIOR = 32 # interior points per method
                # Sobol: must be power of 2 → maximum 32 points with 40 points available
                # LHS: no restriction → 32 chosen to keep sets comparable
N_CORNERS = 4 # corners of the price space [GAS_MIN, GAS_MAX] x [ELEC_MIN, ELEC_MAX] to prevent extrapolation in the surrogate model
N_EDGES = 4
N_TOTAL = N_INTERIOR + N_CORNERS + N_EDGES # = 40

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

def create_df(interior_pts):
    """
    Create a DataFrame with the sampled points and their corresponding labels (corner, edge_midpoint, interior)
    :param interior_pts: Scaled array with margin in [GAS_MIN, GAS_MAX] x [ELEC_MIN, ELEC_MAX]
    :return: DataFrame with columns "gas_price", "electricity_price", "point_type" and the corresponding points as a numpy array
    """
    labels = (["corner"] * N_CORNERS +
              ["edge_midpoint"] * N_EDGES +
              ["interior"] * N_INTERIOR)
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
    :return: Tuple of (discrepancy, minimum pairwise distance)
    """
    unit = qmc.scale(pts,
                     l_bounds=[GAS_MIN,  ELEC_MIN],
                     u_bounds=[GAS_MAX,  ELEC_MAX],
                     reverse=True)
    disc = qmc.discrepancy(unit)
    n = len(pts)
    dists = [np.linalg.norm(pts[i] - pts[j])
             for i in range(n) for j in range(i + 1, n)]
    return disc, min(dists)

def print_summary(name, disc, min_dist):
    """
    Print a summary of the sampling method's quality metrics (discrepancy and minimum pairwise distance)
    :param name: Sampling method's name
    :param disc: Discrepancy of the sampling points, which measures how uniformly the points are distributed in the sample space.
    Lower values indicate better uniformity.
    :param min_dist: Minimum pairwise distance of the sampling points.
    """
    print(f"{name:<6} \n- discrepancy = {disc:.5f}\n- min pairwise dist = {min_dist:.2f} €/MWh")


# == Sampling methods ===========================
def sobol_sequence():
    """
    Generate sampling points using the Sobol sequence method, which is a low-discrepancy sequence
    that provides better coverage of the sample space compared to random sampling.
    :return: DataFrame with the sampling points and quality metrics (discrepancy and minimum pairwise distance) of the sampling method
    """
    sobol_sampler = qmc.Sobol(d=2, scramble=True, seed=28)
    sobol_raw = sobol_sampler.random(N_INTERIOR) # [0,1]², power-of-2 n
    sobol_interior = scale_to_price_domain(sobol_raw)

    df_sobol, sobol_all = create_df(sobol_interior)
    sobol_disc, sobol_min_dist = quality(sobol_all)

    return df_sobol, sobol_disc, sobol_min_dist

def lhs_method():
    """
    Generate sampling points using the Latin Hypercube Sampling (LHS) method, which is a stratified sampling technique
    that ensures that the sample points are well distributed across the sample space.
    :return: DataFrame with the sampling points and quality metrics (discrepancy and minimum pairwise distance) of the sampling method
    """
    lhs_sampler = qmc.LatinHypercube(d=2, scramble=True, optimization="random-cd", seed=28)
    lhs_raw = lhs_sampler.random(N_INTERIOR) # [0,1]², any n works
    lhs_interior = scale_to_price_domain(lhs_raw)

    df_lhs, lhs_all = create_df(lhs_interior)
    lhs_disc, lhs_min_dist = quality(lhs_all)

    return df_lhs, lhs_disc, lhs_min_dist


