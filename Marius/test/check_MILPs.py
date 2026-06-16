"""
This test checks whether the normalized objective function produces
the same OPEX values as the original formulation, within the specified MIP gap.

This is theoretically guaranteed for the optimal solution of the MILP, however because
B&B yields suboptimal solutions within the MIP gap, it is possible that the
solutions may be slightly different.
"""

import sys
sys.path.append("Marius")  # to import formulation_MILP from parent directory

from formulation_MILP import solve

import numpy as np

c_el = np.array([0.1, 0.2, 0.5, 1.0])
c_G = np.array([0.1, 0.2, 0.5, 1.0])
PRICE_PERMUTATIONS = [(c_G_i, c_el_j) for c_G_i in c_G for c_el_j in c_el]

for pair in PRICE_PERMUTATIONS:
    MIP_GAP = 1e-2
    c_G, c_el = pair
    print(f"c_G={c_G:.4f}  c_el={c_el:.4f}")
    opex, _ = solve(c_G, c_el, mip_gap=MIP_GAP)
    print(f"  MILP:      OPEX = {opex:,.2f}")
    opex_norm, _ = solve(c_G, c_el, mip_gap=MIP_GAP, normalize=True)
    print(f"  MILP_norm: OPEX = {opex_norm:,.2f}")

    # Solutions are allowed to be 1 MIP-Gap apart. This is due to
    # the B&B search tree potentially exploring different branches
    # in the two formulations
    print(f"  Relative difference: {(opex_norm-opex)/opex:.2%}\n")
    if abs(opex_norm-opex) > MIP_GAP*opex:
        raise ValueError("OPEX values differ by more than the MIP gap!")
    else:
        print("  [PASS]: OPEX values are within the MIP gap.\n")
